import numpy as np
import emda.core as core
import fcodes2
import emda2.core as core2
from emda2.core import iotools
import warnings
from emda2.ext.overlay import get_avg_fsc

def get_rgi(f):
    myrgi_real = core2.maptools.interp_rgi(data=np.real(f))
    myrgi_imag = core2.maptools.interp_rgi(data=np.imag(f))
    return myrgi_real, myrgi_imag

def get_f(f, myrgi_r, myrgi_i, rotmat):
    nx, ny, nz = f.shape
    points = core2.maptools.get_points(rotmat,nx,ny,nz)
    f_r = myrgi_r(points).reshape(nx,ny,nz)
    f_i = myrgi_i(points).reshape(nx,ny,nz)
    return f_r + 1j * f_i

class TookTooLong(Warning):
    pass

class MinimizeStopper(object):
    def __init__(self):
        self.nit = 0

    def __call__(self, xk):
        self.nit += 1
        if self.nit > 10:
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)
        else:
            print("iter: %i " % self.nit)


def get_dfs(mapin, xyz, vol):
    nx, ny, nz = mapin.shape
    dfs = np.zeros(shape=(nx, ny, nz, 3), dtype=np.complex64)
    for i in range(3):
        dfs[:, :, :, i] = np.fft.fftshift(
            (1 / vol) * 2j * np.pi * np.fft.fftn(mapin * xyz[i])
        )
    return dfs

def create_xyz_grid(nxyz):
    x = np.fft.fftshift(np.fft.fftfreq(nxyz[0]))
    y = np.fft.fftshift(np.fft.fftfreq(nxyz[1]))
    z = np.fft.fftshift(np.fft.fftfreq(nxyz[2]))
    xv, yv, zv = np.meshgrid(x, y, z)
    return [yv, xv, zv]

def get_dfs2(ert, xyz):
    rho = np.real(np.fft.ifftn(np.fft.ifftshift(ert)))
    nx, ny, nz = ert.shape
    #xyz = create_xyz_grid([nx, ny, nz]) <- this routine is
    # not used for speed purpose. precalculated grid is used.
    # otherwise, it is correct. then do not use 1/nx to
    # calculated the const below.
    dfs = np.zeros(shape=(nx, ny, nz, 3), dtype=np.complex64)
    #const = 2j * np.pi #* (1/nx)
    const = 2j * np.pi / 3.0
    for i in range(3):
        dfs[:, :, :, i] = np.fft.fftshift( const * np.fft.fftn(rho * xyz[i]))
    return dfs

class Bfgs:
    def __init__(self):
        self.method = "BFGS"
        self.e0 = None
        self.e1 = None
        self.f1 = None
        self.ert = self.e1
        self.ereal_rgi = None
        self.eimag_rgi = None
        self.wgrid = None
        self.sv = None
        self.q = np.array([1., 0., 0., 0.], 'float')
        self.t = np.array([0., 0., 0.], 'float')
        self.xyz = None
        self.xyz_sum = None
        self.bin_idx = None
        self.nbin = None
        self.binfsc = None
        self.avgfsc = None
        self.step = None
        self.q_prev = None
        self.pixsize = None
        self.x = np.array([0.1, 0.1, 0.1], 'float')
        self.map_dim = None
        self.nit = 0

    def hess_r(self, x):
        from emda.core.quaternions import derivatives_wrt_q
        # rotation derivatives
        tp2 = (2.0 * np.pi) ** 2
        nx, ny, nz = self.e0.shape
        vol = nx * ny * nz

        step = np.asarray(x[3:], 'float')
        q = np.array([1., 0., 0., 0.], 'float') + np.insert(step, 0, 0.0)
        q = q / np.sqrt(np.dot(q, q))
        rotmat = core.quaternions.get_RM(q)
        #ert = maputils.get_FRS(rotmat, self.e1, interp="linear")[:, :, :, 0]
        ert = get_f(self.e1, self.ereal_rgi, self.eimag_rgi, rotmat)

        st, _, _, _ = fcodes2.get_st(nx, ny, nz, x[:3])
        ert = ert * st

        binfsc, _, _ = core.fsc.anytwomaps_fsc_covariance(
            self.e0, ert, self.bin_idx, self.nbin)
        val_arr = np.zeros((self.nbin, 2), dtype='float')
        val_arr[:,0] = binfsc #/ (1.0 - binfsc**2)
        wgrid = fcodes2.read_into_grid2(self.bin_idx,
            val_arr, self.nbin, nz, ny, nx)[:,:,:,0]

        dRdq = derivatives_wrt_q(q)
        ddf = np.zeros((6, 6), dtype="float")
        wfsc = wgrid * np.real(np.conjugate(self.e0) * ert)
        for i in range(3):
            for j in range(3):
                if i == 0 or (i > 0 and j >= i):
                    ddf[i,j] = -tp2 * np.sum(wgrid * self.sv[i,:,:,:] * self.sv[j,:,:,:]) / vol
                    b = np.zeros((3, 3), dtype="float")
                    n = -1
                    for k in range(3):
                        for l in range(3):
                            if k == 0 or (k > 0 and l >= k):
                                n += 1
                                b[k, l] = (
                                    (-tp2 / vol)
                                    * self.xyz_sum[n]
                                    * np.sum(
                                        wfsc
                                        * self.sv[k, :, :, :]
                                        * self.sv[l, :, :, :]
                                        * dRdq[i, k, l]
                                        * dRdq[j, k, l]
                                    )
                                )
                            else:
                                b[k, l] = b[l, k]
                    ddf[i+3, j+3] = np.sum(b) / vol # divide by vol is to scale
                else:
                    ddf[i, j] = ddf[j, i]
                    ddf[i+3, j+3] = ddf[j+3, i+3]
        return ddf

    def derivatives(self, x):
        from emda.core.quaternions import derivatives_wrt_q
        #from emda2.ext.overlay import get_dfs
        # rotation derivatives
        nx, ny, nz = self.e0.shape
        vol = nx * ny * nz
        tpi = (2.0 * np.pi * 1j)
        tp = 2.0 * np.pi

        sv = [self.sv[0,:,:,:], self.sv[1,:,:,:], self.sv[2,:,:,:]]
        dFRS = get_dfs2(self.ert, self.xyz)
        #temp = np.gradient(self.ert)
        #dFRS = np.zeros((nx, ny, nz, 3), 'complex')
        #for i in range(3):
        #    dFRS[:,:,:,i] = temp[i]
            #print('sum: ', np.sum(temp[i]))

        dRdq = derivatives_wrt_q(self.q)
        #print('dRdq')
        #print(dRdq)
        df = np.zeros(6, dtype="float")
        wgrid = self.wgrid #1.0
        for i in range(3):
            df[i] = -np.sum(np.real(wgrid * np.conjugate(self.e0) * (self.ert * tpi * self.sv[i,:,:,:]))) / vol
            a = np.zeros((3, 3), dtype="float")
            for k in range(3):
                for l in range(3):
                    if k == 0 or (k > 0 and l >= k):
                        a[k, l] = np.sum(
                            wgrid
                            * np.real(
                                np.conjugate(self.e0)
                                * (dFRS[:, :, :, k] * self.sv[l, :, :, :] * dRdq[i, k, l])
                                #* (dFRS[:, :, :, k] * self.sv[l, :, :, :] * tp * dRdq[i, k, l])
                            )
                        ) 
                    else:
                        a[k, l] = a[l, k]
            df[i+3] = np.sum(a) / vol # divide by vol is to scale
        #print('derivatives: ')
        #for i in range(6):
        #    print(df[i])
        return df

    def calc_fsc(self):
        assert self.e0.shape == self.e1.shape == self.bin_idx.shape
        binfsc, _, bincounts = core.fsc.anytwomaps_fsc_covariance(
            self.e0, self.ert, self.bin_idx, self.nbin)
        # mask all NaNs
        mask = np.isnan(binfsc)
        binfsc = np.where(~mask,binfsc,0.)
        self.binfsc = binfsc
        self.avgfsc = get_avg_fsc(binfsc=binfsc, bincounts=bincounts)

    def get_wght(self): 
        nz, ny, nx = self.e0.shape
        val_arr = np.zeros((self.nbin, 2), dtype='float')
        val_arr[:,0] = self.binfsc #/ (1 - self.binfsc ** 2)
        self.wgrid = fcodes2.read_into_grid2(self.bin_idx,
            val_arr, self.nbin, nz, ny, nx)[:,:,:,0]

    #3. functional
    def functional(self, x):
        nx, ny, nz = self.e1.shape
        step = np.asarray(x[3:], 'float')
        self.q = np.array([1., 0., 0., 0.], 'float') + np.insert(step, 0, 0.0)
        self.q = self.q / np.sqrt(np.dot(self.q, self.q))
        #print('print self.q: ', self.q)
        rotmat = core.quaternions.get_RM(self.q)
        #ert = maputils.get_FRS(rotmat, self.e1, interp="linear")[:, :, :, 0] # faster, less accurate
        ert = get_f(self.e1, self.ereal_rgi, self.eimag_rgi, rotmat) # slower
        self.t = x[:3]
        self.ert = ert * fcodes2.get_st(nx, ny, nz, self.t)[0]
        self.calc_fsc()
        #print(self.binfsc)
        self.get_wght()
        fval = np.sum(self.wgrid * self.e0 * np.conjugate(self.ert)) / (nx*ny*nz) # divide by vol is to scale
        # print values on display
        rotation = np.arccos((np.trace(rotmat) - 1) / 2) * 180.0 / np.pi
        t_angstrom = self.t * self.pixsize * np.asarray(self.map_dim, 'float')
        translation = np.sqrt(np.dot(t_angstrom, t_angstrom))
        print("fval, rot(deg), trans(A): {:6.5f} {:6.4f} {:6.4f}".format(
                    fval.real, rotation, translation))
        return -fval.real

    #3. optimize using BFGS
    def optimize(self):
        from scipy.optimize import minimize
        self.ereal_rgi, self.eimag_rgi = get_rgi(self.e1)
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'float')
        if self.map_dim is None:
            self.map_dim = self.e1.shape
        nx, ny, nz = self.e1.shape
        self.xyz = create_xyz_grid(self.e1.shape)
        vol = nx * ny * nz
        tol = 1e-4#vol * 1e-9
        print('vol= ', vol)
        print('tol= ', tol)
        minimize_stopper = MinimizeStopper()
        options = {'maxiter': 100, 'gtol': tol}
        print("Optimization method: ", self.method)
        if self.method.lower() == 'nelder-mead':
            result = minimize(
                fun=self.functional, 
                x0=x, 
                method='Nelder-Mead', 
                tol=tol, 
                options=options
                )
        elif self.method.lower() == 'bfgs':
            result = minimize(
                fun=self.functional, 
                x0=x, 
                method='BFGS', 
                jac=self.derivatives, 
                tol=tol, 
                #callback=minimize_stopper.__call__,
                options=options
                )
        else:
            result = minimize(
                fun=self.functional, 
                x0=x, 
                method=self.method, 
                jac=self.derivatives, 
                hess=self.hess_r, 
                tol=tol, 
                options=options
                )    
        if result.status:
            print(result)
        print("Final q: ", self.q)  
        print("Final t: ", self.t)
        print("Euler angles: ", np.rad2deg(core.quaternions.rotationMatrixToEulerAngles(core.quaternions.get_RM(self.q))))     
        print("Outputting fitted map...")
        rotmat = core.quaternions.get_RM(self.q)
        ert = get_f(self.e1, self.ereal_rgi, self.eimag_rgi, rotmat)
        nx, ny, nz = ert.shape
        self.ert = ert * fcodes2.get_st(nx, ny, nz, self.t)[0]



class Bfgs_trans:
    def __init__(self):
        self.e0 = None
        self.e1 = None
        self.ert = self.e1
        self.ereal_rgi = None
        self.eimag_rgi = None
        self.wgrid = None
        self.sv = None
        self.q = np.array([1., 0., 0., 0.], 'float')
        self.t = np.array([0., 0., 0.], 'float')
        self.xyz = None
        self.xyz_sum = None
        self.bin_idx = None
        self.nbin = None
        self.binfsc = None
        self.avgfsc = None
        self.step = None
        self.q_prev = None
        self.x = np.array([0.1, 0.1, 0.1], 'float')

    def hess_r(self, x):
        from emda.core.quaternions import derivatives_wrt_q
        # rotation derivatives
        tp2 = (2.0 * np.pi) ** 2
        nx, ny, nz = self.e0.shape
        vol = nx * ny * nz

        st, _, _, _ = fcodes2.get_st(nx, ny, nz, x)
        ert = self.e1 * st

        binfsc, _, _ = core.fsc.anytwomaps_fsc_covariance(
            self.e0, ert, self.bin_idx, self.nbin)
        val_arr = np.zeros((self.nbin, 2), dtype='float')
        val_arr[:,0] = binfsc #/ (1.0 - binfsc**2)
        wgrid = fcodes2.read_into_grid2(self.bin_idx,
            val_arr, self.nbin, nz, ny, nx)[:,:,:,0]

        ddf = np.zeros((3, 3), dtype="float")
        for i in range(3):
            for j in range(3):
                if i == 0 or (i > 0 and j >= i):
                    ddf[i,j] = -tp2 * np.sum(wgrid * self.sv[i,:,:,:] * self.sv[j,:,:,:]) / vol
                else:
                    ddf[i, j] = ddf[j, i]
        return -ddf

    def derivatives(self, x):
        # translation derivatives
        nx, ny, nz = self.e0.shape
        vol = nx * ny * nz
        tpi = (2.0 * np.pi * 1j)

        st, sv1, sv2, sv3 = fcodes2.get_st(nx, ny, nz, x)
        ert = self.e1 * st
        sv = [sv1, sv2, sv3]
        binfsc, _, _ = core.fsc.anytwomaps_fsc_covariance(
            self.e0, ert, self.bin_idx, self.nbin)
        val_arr = np.zeros((self.nbin, 2), dtype='float')
        val_arr[:,0] = binfsc
        wgrid = fcodes2.read_into_grid2(self.bin_idx,
            val_arr, self.nbin, nz, ny, nx)[:,:,:,0]

        df = np.zeros(3, dtype="float")
        for i in range(3):
            df[i] = np.sum(np.real(wgrid * np.conjugate(self.e0) * (ert * tpi * sv[i]))) / vol
            #df[i] = np.sum(np.real(np.conjugate(self.e0) * (ert * tpi * sv[i]))) / vol
        return -df

    def calc_fsc(self):
        assert self.e0.shape == self.e1.shape == self.bin_idx.shape
        binfsc, _, bincounts = core.fsc.anytwomaps_fsc_covariance(
            self.e0, self.ert, self.bin_idx, self.nbin)
        self.binfsc = binfsc
        self.avgfsc = get_avg_fsc(binfsc=binfsc, bincounts=bincounts)

    def get_wght(self): 
        nz, ny, nx = self.e0.shape
        val_arr = np.zeros((self.nbin, 2), dtype='float')
        val_arr[:,0] = self.binfsc #/ (1 - self.binfsc ** 2)
        self.wgrid = fcodes2.read_into_grid2(self.bin_idx,
            val_arr, self.nbin, nz, ny, nx)[:,:,:,0]

    #3. functional
    def functional(self, x):
        nx, ny, nz = self.e1.shape
        self.t = x
        st, _, _, _ = fcodes2.get_st(nx, ny, nz, self.t)
        self.ert = self.e1 * st
        self.calc_fsc()
        self.get_wght()
        fval = np.sum(self.wgrid * self.e0 * np.conjugate(self.ert)) / (nx*ny*nz) # divide by vol is to scale
        print('fval, t ', fval.real, self.t)
        return -fval.real

    #3. optimize using BFGS
    def optimize(self):
        from scipy.optimize import minimize
        self.ereal_rgi, self.eimag_rgi = get_rgi(self.e1)
        x = np.array([0., 0., 0.], 'float')
        options = {'maxiter': 100}
        #result = minimize(fun=self.functional, x0=x, method='Nelder-Mead') # worked
        result = minimize(fun=self.functional, x0=x, method='Newton-CG', jac=self.derivatives, hess=self.hess_r, tol=1e-5) # worked
        #result = minimize(fun=self.functional, x0=x, method='BFGS', jac=self.derivatives, tol=1e-5, options=options) # worked
        print(result)
        self.t = result.x 
        print('translation vec: ', self.t)    