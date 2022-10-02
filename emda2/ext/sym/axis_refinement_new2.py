from __future__ import absolute_import, division, print_function, unicode_literals
import traceback
import numpy as np
import math
from numpy.fft import fftn, ifftn, fftshift, ifftshift
# EMDA2 imports
from emda2.core import restools, plotter, iotools, fsctools
from emda2.ext.utils import (
    rotate_f, 
    shift_density, 
    center_of_mass_density, 
    cut_resolution_for_linefit,
    determine_ibin,
    get_ibin,
    filter_fsc,
    vec2string
)
import fcodes2 as fc
from emda2.core import fsctools

def vec2string(vec):
    return " ".join(("% .3f" % x for x in vec))

def ax2spherical(ax):
    from math import sqrt, acos, atan2
    ax = np.asarray(ax, 'float')
    ax = ax / sqrt(np.dot(ax, ax))
    Ux, Uy, Uz = list(ax)
    phi = atan2(Uy, Ux)
    thet = acos(Uz)
    return phi, thet

def spherical2ax(phi, thet):
    # phi, thet in Radians
    from math import cos, sin
    Ux = cos(phi)*sin(thet)
    Uy = sin(phi)*sin(thet)
    Uz = cos(thet)
    return [Ux, Uy, Uz]

def rotmat_spherical_crd(phi, thet, angle):
    # all angle must be fed in Radians
    from math import cos, sin, acos, atan2, sqrt

    # Rotation from axis-angle using Rodriguez formula
    # R = cos(om)*I + sin(om)*a + (1-cos(om))*a^2
    om = float(angle)
    #om = np.deg2rad(angle)
    s1 = sin(om)
    c1 = (1-cos(om))

    tm1 = cos(om) * np.identity(3)

    tm2 = np.array([[ 0,                  -cos(thet),          sin(phi)*sin(thet)],
                    [ cos(thet),           0,                 -cos(phi)*sin(thet)],
                    [-sin(phi)*sin(thet),  cos(phi)*sin(thet), 0]
                    ], 'float')

    a11 = cos(phi)**2 * sin(thet)**2
    a12 = cos(phi) * sin(phi) * sin(thet)**2
    a13 = cos(phi) * sin(thet) * cos(thet)
    a21 = a12
    a22 = sin(phi)**2 * sin(thet)**2
    a23 = sin(phi) * sin(thet) * cos(thet)
    a31 = a13
    a32 = a23
    a33 = cos(thet)**2

    tm3 = np.array([[a11, a12, a13],
                    [a21, a22, a23],
                    [a31, a32, a33]], 'float')

    R = tm1 + s1 * tm2 + c1 * tm3

    # DERIVATIVES of rotation matrix w.r.t. phi and theta
    # using thrigonometric identities - 
    # (cos(x))^2 = (1 + cos(2x))/2
    # (sin(x)^2) = (1 - cos(2x))/2
    # cos(x)*sin(x) = (sin(2x))/2

    # derivatives of tm2 w.r.t phi
    dtm2_dp11 = 0.
    dtm2_dp12 = 0.
    dtm2_dp13 = cos(phi)*sin(thet)
    dtm2_dp21 = 0.
    dtm2_dp22 = 0.
    dtm2_dp23 = sin(phi)*sin(thet)
    dtm2_dp31 = -cos(phi)*sin(thet)
    dtm2_dp32 = -sin(phi)*sin(thet)
    dtm2_dp33 = 0.
    dtm2_dp = np.array([
                        [dtm2_dp11, dtm2_dp12, dtm2_dp13],
                        [dtm2_dp21, dtm2_dp22, dtm2_dp23],
                        [dtm2_dp31, dtm2_dp32, dtm2_dp33]
                        ], 'float')

    # derivatives of tm2 w.r.t theta
    dtm2_dt11 = 0.
    dtm2_dt12 = sin(thet)
    dtm2_dt13 = sin(phi)*cos(thet)
    dtm2_dt21 = -sin(thet)
    dtm2_dt22 = 0.
    dtm2_dt23 = -cos(phi)*cos(thet)
    dtm2_dt31 = -sin(phi)*cos(thet)
    dtm2_dt32 = cos(phi)*cos(thet)
    dtm2_dt33 = 0.
    dtm2_dt = np.array([
                        [dtm2_dt11, dtm2_dt12, dtm2_dt13],
                        [dtm2_dt21, dtm2_dt22, dtm2_dt23],
                        [dtm2_dt31, dtm2_dt32, dtm2_dt33]
                        ], 'float')

    # derivatives of tm3 w.r.t phi
    dtm3_dp11 = -0.5 * sin(2*phi)*(1.0 - cos(2*thet))
    dtm3_dp12 = 0.5 * cos(2*phi)*(1.0 - cos(2*thet))
    dtm3_dp13 = -0.5 * sin(2*thet)*sin(phi)
    dtm3_dp21 = dtm3_dp12
    dtm3_dp22 = 0.5 * sin(2*phi)*(1.0-cos(2*thet))
    dtm3_dp23 = 0.5 * sin(2*thet)*cos(phi)
    dtm3_dp31 = dtm3_dp13
    dtm3_dp32 = dtm3_dp23
    dtm3_dp33 = 0.
    dtm3_dp = np.array([
                        [dtm3_dp11, dtm3_dp12, dtm3_dp13],
                        [dtm3_dp21, dtm3_dp22, dtm3_dp23],
                        [dtm3_dp31, dtm3_dp32, dtm3_dp33]
                        ], 'float')

    # derivatives of tm3 w.r.t theta
    dtm3_dt11 = 0.5 * (1.0 + cos(2*phi))*sin(2*thet)
    dtm3_dt12 = 0.5 * sin(2*phi)*sin(2*thet)
    dtm3_dt13 = cos(2*thet)*cos(phi)
    dtm3_dt21 = dtm3_dt12
    dtm3_dt22 = 0.5 * (1.0 - cos(2*phi))*sin(2*thet)
    dtm3_dt23 = cos(2*thet)*sin(phi)
    dtm3_dt31 = dtm3_dt13
    dtm3_dt32 = dtm3_dt23
    dtm3_dt33 = -sin(2*thet)
    dtm3_dt = np.array([
                        [dtm3_dt11, dtm3_dt12, dtm3_dt13],
                        [dtm3_dt21, dtm3_dt22, dtm3_dt23],
                        [dtm3_dt31, dtm3_dt32, dtm3_dt33]
                        ], 'float')

    dR_dp = s1 * dtm2_dp + c1 * dtm3_dp
    dR_dt = s1 * dtm2_dt + c1 * dtm3_dt
    return R, dR_dp, dR_dt

class EmmapOverlay:
    def __init__(self, arr):
        self.arr = arr
        self.map_unit_cell = None
        self.map_dim = arr.shape
        self.com = False
        self.com1 = None
        self.com2 = None
        self.box_centr = None
        self.nbin = None
        self.res_arr = None
        self.bin_idx = None
        self.fo_lst = []
        self.eo_lst = []
        self.pix = None

    def prep_data(self):
        com = center_of_mass_density(self.arr)
        print("com:", com)
        nx, ny, nz = self.arr.shape
        self.box_centr = (nx // 2, ny // 2, nz // 2)
        self.com1 = com
        if self.com:
            arr = shift_density(self.arr, np.subtract(self.box_centr, com))
            self.com2 = center_of_mass_density(arr)
            print("com after centering:", self.com2)
            self.fo_lst.append(fftshift(fftn(fftshift(arr))))
        else:
            self.fo_lst.append(fftshift(fftn(fftshift(self.arr))))
        if any(itm is None for itm in [self.bin_idx, self.nbin, self.res_arr]):
            if self.map_unit_cell is not None:
                self.pix = [self.map_unit_cell[i]/sh for i, sh in enumerate(self.map_dim)]
                self.nbin, self.res_arr, self.bin_idx, _ = restools.get_resolution_array(
                    self.map_unit_cell, self.fo_lst[0]
                )
            else:
                print("make sure cell is included")


def create_xyz_grid(nxyz):
    x = np.fft.fftshift(np.fft.fftfreq(nxyz[0]))
    y = np.fft.fftshift(np.fft.fftfreq(nxyz[1]))
    z = np.fft.fftshift(np.fft.fftfreq(nxyz[2]))
    xv, yv, zv = np.meshgrid(x, y, z)
    return [yv, xv, zv]

def create_xyz_grid2(nxyz):
    x = np.fft.fftshift(np.fft.fftfreq(nxyz[0], 1/nxyz[0]))
    y = np.fft.fftshift(np.fft.fftfreq(nxyz[1], 1/nxyz[1]))
    z = np.fft.fftshift(np.fft.fftfreq(nxyz[2], 1/nxyz[2]))
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
        self.ax_init = None
        self.ax_final = None
        self.phi = None
        self.thet = None
        self.phi0 = None
        self.thet0 = None
        self.pixsize = None
        self.angle = None
        self.q = np.array([1., 0., 0., 0.], 'float')
        self.t_init = None #np.array([0., 0., 0.], 'float')
        self.t = None
        self.xyz = None
        self.xyz_sum = None
        self.vol = None
        self.bin_idx = None
        self.binfsc = None
        self.nbin = None
        self.fobj = None

    def calc_fsc(self):
        assert self.e0.shape == self.e1.shape == self.bin_idx.shape
        binfsc, _, bincounts = fsctools.anytwomaps_fsc_covariance(
            self.e0, self.e1, self.bin_idx, self.nbin)
        self.binfsc = binfsc

    def get_wght(self): 
        nz, ny, nx = self.e0.shape
        val_arr = np.zeros((self.nbin, 2), dtype='float')
        val_arr[:,0] = self.binfsc #/ (1 - self.binfsc ** 2)
        self.wgrid = fc.read_into_grid2(self.bin_idx,
            val_arr, self.nbin, nz, ny, nx)[:,:,:,0]

    def numeric_deri(self, x, info):
        # 1st and 2nd derivative from finite difference
        from emda2.ext.bfgs import get_f
        wgrid = 1.0
        nx, ny, nz = self.e0.shape
        df = np.zeros(5, dtype="float") # Jacobian
        self.ddf = np.zeros((5,5), dtype="float") # Hessian
        tpi = (2.0 * np.pi * 1j)
        df = np.zeros(5, dtype="float")
        for i in range(3):
            df[i] = -np.sum(np.real(wgrid * np.conjugate(self.e0) * (self.e1 * tpi * self.sv[i,:,:,:])))
        dphi = 1/(2*nx)
        rotmat_f, _, _ = rotmat_spherical_crd(phi=self.phi+dphi, thet=self.thet, angle=self.angle)
        rotmat_b, _, _ = rotmat_spherical_crd(phi=self.phi-dphi, thet=self.thet, angle=self.angle)
        e1f_dphi = self.st * get_f(self.e0, self.ereal_rgi, self.eimag_rgi, rotmat_f)
        e1b_dphi = self.st * get_f(self.e0, self.ereal_rgi, self.eimag_rgi, rotmat_b)
        dFdphi = (e1f_dphi - e1b_dphi) / (2*dphi)
        df[3] = -np.sum(wgrid * np.real(self.e0 * np.conjugate(dFdphi)))
        ddFddphi = (e1b_dphi - 2*self.e1 + e1f_dphi) / (dphi**2)
        self.ddf[3,3] = -np.sum(wgrid * np.real(self.e0 * np.conjugate(ddFddphi)))
        dthet  = 1/(2*nx)
        rotmat_f, _, _ = rotmat_spherical_crd(phi=self.phi, thet=self.thet+dthet, angle=self.angle)
        rotmat_b, _, _ = rotmat_spherical_crd(phi=self.phi, thet=self.thet-dthet, angle=self.angle)
        e1f_dthet = self.st * get_f(self.e0, self.ereal_rgi, self.eimag_rgi, rotmat_f)
        e1b_dthet = self.st * get_f(self.e0, self.ereal_rgi, self.eimag_rgi, rotmat_b)
        dFdthet = (e1f_dthet - e1b_dthet) / (2*dthet)
        df[4] = -np.sum(wgrid * np.real(self.e0 * np.conjugate(dFdthet)))
        ddFddthet = (e1b_dthet - 2*self.e1 + e1f_dthet) / (dthet**2)
        self.ddf[4,4] = -np.sum(wgrid * np.real(self.e0 * np.conjugate(ddFddthet))) 
        return df
    
    def secondd(self, x, info):
        # 2nd derivative from finite difference
        wgrid = 1.0
        tp2 = (2.0 * np.pi)**2
        nx, ny, nz = self.e0.shape
        ddf = np.zeros((5,5), dtype="float")
        for i in range(3):
            for j in range(3):
                ddf[i,j] = -tp2 * np.sum(wgrid * self.sv[i,:,:,:] * self.sv[j,:,:,:]) 
        ddf[3,3] = self.ddf[3,3]
        ddf[4,4] = self.ddf[4,4]
        return ddf


    def numeric_derivative(self, x, info):
        # only 1st derivative
        from emda2.ext.bfgs import get_f
        wgrid = 1.0
        nx, ny, nz = self.e0.shape
        vol = nx * ny * nz
        tpi = (2.0 * np.pi * 1j)
        df = np.zeros(5, dtype="float")
        for i in range(3):
            df[i] = -np.sum(np.real(wgrid * np.conjugate(self.e0) * (self.e1 * tpi * self.sv[i,:,:,:]))) #/ vol
        dphi = 1/(2*nx) #0.01
        phi = self.phi + dphi
        rotmat, _, _ = rotmat_spherical_crd(phi=phi, thet=self.thet, angle=self.angle)
        e1_dphi = self.st * get_f(self.e0, self.ereal_rgi, self.eimag_rgi, rotmat)
        dFdphi = (e1_dphi - self.e1) / dphi
        df[3] = -np.sum(wgrid * np.real(self.e0 * np.conjugate(dFdphi)))
        dthet  = 1/(2*nx) #0.01
        thet = self.thet + dthet
        rotmat, _, _ = rotmat_spherical_crd(phi=self.phi, thet=thet, angle=self.angle)
        e1_dthet = self.st * get_f(self.e0, self.ereal_rgi, self.eimag_rgi, rotmat)
        dFdthet = (e1_dthet - self.e1) / dthet
        df[4] = -np.sum(wgrid * np.real(self.e0 * np.conjugate(dFdthet)))
        print('df', df)
        #print('step: ', dphi*df[3], dthet*df[4])
        return df

    def derivatives(self, x, info):
        # analytical solution- only 1st derivative
        nx, ny, nz = self.e0.shape
        vol = nx * ny * nz
        tpi = (2.0 * np.pi * 1j)
        ert = self.e1
        temp = np.gradient(ert)
        dFRS = np.zeros((nx, ny, nz, 3), 'complex')
        for i in range(3):
            dFRS[:,:,:,i] = temp[i]
        #dFRS = get_dfs2(ert, self.xyz)
        _, dR_dphi, dR_dtheta = rotmat_spherical_crd(phi=self.phi, thet=self.thet, angle=self.angle)
        #print()
        #print(dR_dphi)
        #print()
        #print(dR_dtheta)
        #print()
        df = np.zeros(5, dtype="float")
        wgrid = 1.0
        #wgrid = self.wgrid#1.0
        for i in range(3):
            df[i] = -np.sum(np.real(wgrid * np.conjugate(self.e0) * (ert * tpi * self.sv[i,:,:,:]))) #/ vol
        # using phi and theta spherical coordinates
        a = np.zeros((3, 3), dtype="float")
        b = np.zeros((3, 3), dtype="float")
        for k in range(3):
            for l in range(3):
                """ if k == 0 or (k > 0 and l >= k):
                    x1 = dFRS[:, :, :, k] * self.sv[l, :, :, :]
                    temp1 = x1 * dR_dphi[k, l]
                    a[k, l] = np.sum(wgrid * np.real(self.e0 * np.conjugate(temp1)))
                    temp2 = x1 * dR_dtheta[k, l]
                    b[k, l] = np.sum(wgrid * np.real(self.e0 * np.conjugate(temp2)))
                else:
                    a[l, k] = a[k, l]
                    b[l, k] = b[k, l] """
                x1 = dFRS[:, :, :, k] * self.sv[l, :, :, :]
                temp1 = x1 * dR_dphi[k, l]
                a[k, l] = np.sum(wgrid * np.real(self.e0 * np.conjugate(temp1)))
                temp2 = x1 * dR_dtheta[k, l]
                b[k, l] = np.sum(wgrid * np.real(self.e0 * np.conjugate(temp2)))                
        #print()
        #print(a)
        #print()
        #print(b)
        #print()
        df[3] = -np.sum(a) #/ vol
        df[4] = -np.sum(b) #/ vol
        print('df: ', df)
        return df

    def callback(self, x, info):
        #if info['Nfeval'] == 5: 
        #    print(x)
        #    return True
        pass

    def functional(self, x, info):
        from emda2.ext.bfgs import get_f
        nx, ny, nz = self.e0.shape
        t = np.asarray(self.t_init, 'float') + x[:3]
        self.st = fc.get_st(nx, ny, nz, t)[0]
        dphi, dthet = x[3], x[4]
        self.phi = self.phi0 + dphi
        self.thet = self.thet0 + dthet
        self.rotmat, _, _ = rotmat_spherical_crd(phi=self.phi, thet=self.thet, angle=self.angle)
        ax = spherical2ax(phi=self.phi, thet=self.thet)
        self.e1 = self.st * get_f(self.e0, self.ereal_rgi, self.eimag_rgi, self.rotmat)
        self.calc_fsc()
        self.get_wght()
        fval = np.sum(self.wgrid * self.e0 * np.conjugate(self.e1))
        tt = [nx*t[0], ny*t[1], nz*t[2]]
        t2 = np.asarray(tt, 'float') * np.asarray(self.pixsize)
        afsc = np.average(self.binfsc)
        print('cycle, fval, afsc, ax, trans: ', info['Nfeval'], fval.real, afsc, ax, t2)
        self.fobj.write('cycle=%i, fval=%.3f, afsc=%.3f, ax=%s, trans=%s\n' %(
            info['Nfeval'], fval.real, afsc, vec2string(ax), vec2string(t2)))
        info['Nfeval'] += 1
        return -fval.real

    def optimize(self):
        from scipy.optimize import minimize
        from emda2.ext.bfgs import get_rgi
        nx, ny, nz = self.e0.shape
        self.xyz = create_xyz_grid(self.e0.shape)
        grid2 = create_xyz_grid2(self.e0.shape)
        self.sv = np.stack(grid2, axis = 0)
        tol = 1e-4
        gtol = 1e-4
        t = np.array([0., 0., 0.], 'float')
        #_, s1, s2, s3 = fc.get_st(nx, ny, nz, t)
        #self.sv = np.array([s1, s2, s3])
        self.ereal_rgi, self.eimag_rgi = get_rgi(self.e0)
        print('initial values of phi, theta, angle:')
        print(self.phi0, self.thet0, self.angle)
        x = np.array([0., 0., 0., 0., 0.], 'float')
        #options = {'maxiter':2000, 'gtol':gtol}
        options = {'maxiter':2000}
        args=({'Nfeval':0},)
        print("Optimization method: ", self.method)
        self.fobj.write("Optimization method: %s\n" %self.method)
        if self.method.lower() == 'bfgs':
            result = minimize(
                fun=self.functional, 
                x0=x, 
                method=self.method,
                #method='bfgs',
                #method='SLSQP',
                #method='Newton-CG',
                #method='TNC',
                #method='COBYLA', 
                #method='trust-constr',
                #method='dogleg',
                #method='trust-ncg',
                #method='trust-exact',
                #method='nelder-mead',
                #jac=self.derivatives,
                #jac=self.numeric_derivative,
                jac=self.numeric_deri,
                hess=self.secondd,
                tol=tol,
                #bounds=[(-1,1), (-1,1), (-1,1), (-np.pi,np.pi), (0,np.pi)], 
                options=options,  
                args=args,
                #callback=self.callback,
                )
        self.fobj.write('\n')
        if result.status:
            print(result)
            self.fobj.write('%s' % result)
        self.fobj.write('\n')
        self.t = np.asarray(self.t_init, 'float') + result.x[:3]
        self.phi = self.phi0 + result.x[3]
        self.thet = self.thet0 + result.x[4] 


    def functional2(self, x):
        from emda2.ext.bfgs import get_f
        nx, ny, nz = self.e0.shape
        t = np.asarray(self.t_init, 'float') + x[:3]
        self.st = fc.get_st(nx, ny, nz, t)[0]
        dphi, dthet = x[3], x[4]
        self.phi = dphi
        self.thet = dthet
        self.rotmat, _, _ = rotmat_spherical_crd(phi=self.phi, thet=self.thet, angle=self.angle)
        ax = spherical2ax(phi=self.phi, thet=self.thet)
        self.e1 = self.st * get_f(self.e0, self.ereal_rgi, self.eimag_rgi, self.rotmat)
        self.calc_fsc()
        self.get_wght()
        fval = np.sum(self.wgrid * self.e0 * np.conjugate(self.e1)) #/ (nx*ny*nz) # divide by vol is to scale
        tt = [nx*t[0], ny*t[1], nz*t[2]]
        t2 = np.asarray(tt, 'float') * np.asarray(self.pixsize)
        afsc = np.average(self.binfsc)
        print('fval, afsc, ax, trans: ', fval.real, afsc, ax, t2)
        self.fobj.write('fval=%.3f, afsc=%.3f, ax=%s, trans=%s\n' %(
            fval.real, afsc, vec2string(ax), vec2string(t2)))
        return -fval.real

    def differential_evol(self):
        from scipy.optimize import rosen, differential_evolution
        from emda2.ext.bfgs import get_rgi
        tol = 1e-4
        self.ereal_rgi, self.eimag_rgi = get_rgi(self.e0)
        print('initial values of phi, theta, angle:')
        print(self.phi0, self.thet0, self.angle)
        x = np.array([0., 0., 0., self.phi0, self.thet0], 'float')
        options = {'maxiter':2000}
        args=({'Nfeval':0},)
        print("Optimization method: ", self.method)
        self.fobj.write("Optimization method: %s\n" %self.method)
        if self.method.lower() == 'bfgs':
            bounds = [(-1,1), (-1,1), (-1,1), (-np.pi,np.pi), (0,np.pi)]
            result = differential_evolution(
                func=self.functional2, 
                bounds=bounds, 
                tol=tol, 
                x0=x 
                )
        self.fobj.write('\n')
        if result.status:
            print(result)
            self.fobj.write('%s' % result)
        self.fobj.write('\n')
        self.t = np.asarray(self.t_init, 'float') + result.x[:3]
        self.phi = self.phi0 + result.x[3]
        self.thet = self.thet0 + result.x[4]         


def fsc_between_static_and_transfomed_map(emmap1, rm, t):
    fo=emmap1.fo_lst[0]
    bin_idx=emmap1.bin_idx
    nbin=emmap1.nbin
    mask=emmap1.mask
    fo = emmap1.fo_lst[0]
    nx, ny, nz = fo.shape
    st, _, _, _ = fc.get_st(nx, ny, nz, t)
    frt = st * rotate_f(rm, fo, interp="linear")[:, :, :, 0]
    # apply mask on rotated and translated map to remove aliasing
    map2 = (ifftshift((ifftn(ifftshift(frt))).real)) * mask
    frt = fftshift(fftn(fftshift(map2)))
    f1f2_fsc = fsctools.anytwomaps_fsc_covariance(
        fo, frt, bin_idx, nbin)[0]
    return f1f2_fsc, frt


def run_fit(
    emmap1,
    args,
    fobj=None,
):
    phi = args['phi']
    thet = args['thet']
    angle = args['angle']
    t = args['t_init']
    optmethod = args['optmethod']
    rotmat, _, _ = rotmat_spherical_crd(phi, thet, angle)
    if emmap1.fitres is not None:
        if emmap1.fitres <= emmap1.res_arr[-1]:
            fitbin = len(emmap1.res_arr) - 1
        else:
            dist = np.sqrt((emmap1.res_arr - emmap1.fitres) ** 2)
            ibin = np.argmin(dist)
            if ibin % 2 != 0:
                ibin = ibin - 1
            fitbin = min([len(dist), ibin])
    if emmap1.fitres is None:
        fitbin = len(emmap1.res_arr) - 1
    fsc_lst = []
    is_abandon = False
    final_t = t
    try:
        for i in range(emmap1.ncycles):
            if i == 0:
                f1f2_fsc, frt = fsc_between_static_and_transfomed_map(
                    emmap1=emmap1,
                    rm=rotmat,
                    t=t,
                )
                print('initial rotmat:' )
                print(rotmat)
                print('initial t:')
                print(t)
                print('writing out movingmap_initial.mrc')
                transformedmap = np.real(ifftshift(ifftn(ifftshift(frt))))
                tm1 = iotools.Map('movingmap_initial.mrc')
                tm1.arr = transformedmap
                tm1.cell = emmap1.map_unit_cell
                tm1.origin = [0,0,0]
                tm1.write()
                fsc_lst.append(f1f2_fsc)
                if emmap1.fitfsc > 0.999:
                    print("\n***FSC between static and moving maps***\n")
                    print("bin#     resolution(A)      start-FSC     end-FSC\n")
                    for j in range(len(emmap1.res_arr)):
                        print(
                            "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
                                j, emmap1.res_arr[j], fsc_lst[0][j], fsc_lst[0][j]
                            )
                        )
                    break
                ibin = get_ibin(filter_fsc(f1f2_fsc), cutoff=emmap1.fitfsc)
                ibin_old = ibin
                if ibin >= 5:
                    if fitbin < ibin:
                        ibin = fitbin
                    ibin_old = ibin
                    print("Fitting starts at ", emmap1.res_arr[ibin], " (A)")
                    fobj.write("Fitting starts at %s (A) \n" %(emmap1.res_arr[ibin]))
                else:
                    print('Resolution= %s (A)' %emmap1.res_arr[ibin])
                    print('FSC between two copies is too low. FSC= %s ibin=%s' %(f1f2_fsc[ibin], ibin))
            else:
                f1f2_fsc, frt = fsc_between_static_and_transfomed_map(
                    emmap1=emmap1,
                    rm=rotmat,
                    t=t
                )          
                ibin = get_ibin(filter_fsc(f1f2_fsc), cutoff=emmap1.fitfsc)
                if fitbin < ibin:
                    ibin = fitbin
                if ibin_old == ibin:
                    print('ibin_old, ibin: ')
                    print(ibin_old, ibin)
                    #final_axis = spherical2ax(phi, thet)
                    fsc_lst.append(f1f2_fsc)
                    res_arr = emmap1.res_arr[:ibin]
                    fsc_bef = fsc_lst[0][:ibin]
                    fsc_aft = fsc_lst[1][:ibin]
                    refined_fsc = fsc_lst[1]
                    print("\n***FSC between static and moving maps***\n")
                    print("bin#     resolution(A)      start-FSC     end-FSC\n")
                    for j in range(len(res_arr)):
                        print(
                            "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
                                j, res_arr[j], fsc_bef[j], fsc_aft[j]
                            )
                        )
                    print("Plotting FSCs...")
                    plotter.plot_nlines(
                        res_arr=res_arr, 
                        list_arr=[fsc_lst[0][:ibin_old], fsc_lst[1][:ibin_old]], 
                        curve_label=["Proshade axis", "EMDA axis"], 
                        plot_title="FSC based on Symmetry axis", 
                        fscline=1.,
                        mapname="fsc_axis.eps")
                    stmap = np.real(ifftshift(ifftn(ifftshift(emmap1.fo_lst[0]))))
                    stm = iotools.Map('static_map.mrc')
                    stm.arr = stmap
                    stm.cell = emmap1.map_unit_cell
                    stm.origin = [0,0,0]
                    stm.write()
                    print('final rotmat:' )
                    print(rotmat)
                    print('final t:')
                    print(final_t)
                    print('writing out fitted_map.mrc.mrc')
                    transformedmap = np.real(ifftshift(ifftn(ifftshift(frt))))
                    tm = iotools.Map('fitted_map.mrc')
                    tm.arr = transformedmap * emmap1.mask
                    tm.cell = emmap1.map_unit_cell
                    tm.origin = [0,0,0]
                    tm.write()                    
                    break
                elif ibin_old > ibin:
                    is_abandon = True
                    print('axis refinement went wrong!')
                    print('ibin_old, ibin', ibin_old, ibin)
                    final_axis = []
                    final_t = []
                    pos_ax = []
                    refined_fsc = fsc_lst[0]
                    break
                else:
                    ibin_old = ibin
                    print("Fitting starts at ", emmap1.res_arr[ibin], " (A)")
                    fobj.write("Fitting starts at %s (A) \n" %(emmap1.res_arr[ibin]))
            if ibin >= 5:
                e_list = [emmap1.eo_lst[0]]
                fcut, cBIdx, cbin = cut_resolution_for_linefit(
                    e_list, emmap1.bin_idx, emmap1.res_arr, ibin
                )
                static_cutmap = fcut[0, :, :, :]  # use Eo for fitting.
                bfgs = Bfgs()
                bfgs.fobj = fobj
                bfgs.phi0 = phi
                bfgs.thet0 = thet
                bfgs.angle = angle
                bfgs.t_init = t
                bfgs.e0 = static_cutmap
                bfgs.bin_idx = cBIdx
                bfgs.nbin = cbin
                bfgs.pixsize = emmap1.pix
                bfgs.method = optmethod
                bfgs.optimize()
                #bfgs.differential_evol()
                t = bfgs.t
                phi = bfgs.phi
                thet = bfgs.thet
                rotmat, _, _ = rotmat_spherical_crd(phi, thet, angle)
                final_axis = spherical2ax(phi, thet)
                final_t = t
            else:
                is_abandon = True
                print("ibin = %s, Cannot proceed axis refinement." %(ibin))
                fobj.write("ibin = %s, Cannot proceed axis refinement. \n" %(ibin))
                final_axis = []
                final_t = []
                pos_ax = []
                refined_fsc = fsc_lst[0]
                break
        if not is_abandon:
            if emmap1.com:
                pos_ax = [(emmap1.com1[i] + final_t[i]*emmap1.map_dim[i])*emmap1.pix[i] for i in range(3)]
            else:
                emmap1.com1 = [emmap1.map_dim[i]//2 for i in range(3)]
                pos_ax = [(emmap1.com1[i] + final_t[i]*emmap1.map_dim[i])*emmap1.pix[i] for i in range(3)]
        return [final_axis, final_t, pos_ax, refined_fsc]
    except:
        fobj.write(traceback.format_exc())
        

def axis_refine(
    emmap1,
    rotaxis,
    symorder,
    t_init=[0.0, 0.0, 0.0],
    fobj=None,
    optmethod='bfgs',
):
    if fobj is None:
        fobj = open('EMDA_axis-refinement.txt', 'w')
    axis = np.asarray(rotaxis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    print(" ")
    print("Initial axis and fold: ", axis, symorder)
    print("Number of refinement cycles:", emmap1.ncycles)
    print("Data resolution for refinement: ", emmap1.fitres)
    print("Initial axis and angles:")
    angle = np.deg2rad(float(360.0 / symorder))
    print("   ", axis, angle)
    phi, theta = ax2spherical(axis)
    args = {'phi':phi, 
            'thet':theta, 
            'angle':angle, 
            't_init':np.asarray(t_init, dtype="float"),
            'optmethod':optmethod}
    results = run_fit(
        emmap1=emmap1,
        fobj=fobj,
        args=args,
    )
    return results


if __name__ == "__main__":
    #imap = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-0608/emd_0608.map"
    #imap =  "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-0139/emda_reboxedmap_emd-0139.map.mrc"
    imap = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-0689/emd_0689.map"
    #imap = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-20690/emd_20690.map"
    #imap = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21971/emd_21971_reboxed.map"
    #imap = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-22199/emd_22199.map"
    #imap = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-22473/emda_reboxedmap_emd-22473.map.mrc"

    #axis = [-0.001056105461950766, -0.001088473810292759, 0.9999988499323474]# 0608 # 7-fold

    #axis = [0.005,  0.043,  0.999] #0139 2-fold
    #axis = [0.000,  0.039,  0.999]

    axis = [0.9, 0.1, 0.0] #0689 2-fold ana-dv fialed, numdev-success
    #axis = [0.993, -0.116, -0.003] #0689 2-fold success
    #axis = [0.038,  0.999,  0.000] #0689 2-fold ana, num dev - success
    #axis = [0.05,  0.02,  0.99] #0689 2-fold unseccess ana,num dev-success

    #axis = [+0.998,    -0.053,    -0.023] #20690 2-fold
    #axis  = [+0.003,    +0.021,    +1.000] #20690 2-fold
    #axis = [+0.744,    +0.668,    -0.027] #20690 2-fold
    #axis  = [+0.01,    +0.98,    +0.01] #20690 2-fold
    #axis = [-0.668,    +0.743,    -0.027]

    #axis = [-0.023,    -0.072,    +0.997] #21971 2-fold
    #axis = [-0.023485169901027482, -0.07274379026152454, 0.9970741134805914] #21971 2-fold

    #axis = [+0.002,    -0.000,    +1.000] #22473 2-fold
    #axis = [0.727,  0.687,  0.002] #22473 2-fold
    
    symorder = 2
    m1 = iotools.Map(imap)
    m1.read()
    emmap1 = EmmapOverlay(arr=m1.workarr)
    emmap1.map_unit_cell = m1.workcell
    emmap1.prep_data()
    final_ax, final_t, ax_pos = axis_refine(
        emmap1,
        axis,
        symorder, 
        fitres=8, 
        fitfsc=0.2, 
        optmethod='BFGS',
        )