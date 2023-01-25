"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import traceback
import numpy as np
import math
from emda2.core import emdalogger
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from emda2.core import restools, plotter, iotools, fsctools
from emda2.ext.utils import (
    #rotate_f, 
    shift_density, 
    center_of_mass_density, 
    cut_resolution_for_linefit,
    get_ibin,
    filter_fsc,
    vec2string
)
import fcodes2 as fc
from emda2.core import fsctools
from math import cos, sin, sqrt, acos, atan2
from emda2.ext.bfgs import get_rgi, get_f
from timeit import default_timer as timer

def vec2string(vec):
    return " ".join(("% .3f" % x for x in vec))

def combinestring(vec):
    return "-".join(("%.3f" % x for x in vec))

def plot_fscs():
    # local plot mimicking plotter.plot_nlines
    pass

def rotate_f(rm, f, bin_idx, ibin):
    if len(f.shape) == 3:
        f = np.expand_dims(f, axis=3)
    nx, ny, nz, ncopies = f.shape
    frs = fc.trilinear_sphere(rm,f,bin_idx,0,ibin,ncopies,nx,ny,nz)
    return frs

def ax2spherical(ax):
    ax = np.asarray(ax, 'float')
    ax = ax / sqrt(np.dot(ax, ax))
    Ux, Uy, Uz = list(ax)
    phi = atan2(Uy, Ux)
    thet = acos(Uz)
    return phi, thet

def spherical2ax(phi, thet):
    # phi, thet in Radians
    Ux = cos(phi)*sin(thet)
    Uy = sin(phi)*sin(thet)
    Uz = cos(thet)
    return [Ux, Uy, Uz]

def rotmat_spherical_crd(phi, thet, angle):
    # all angle must be fed in Radians
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
        self.st = None
        self.mask = None
        self.ax_init = None
        self.ax_final = None
        self.phi = None
        self.thet = None
        self.phi0 = None
        self.thet0 = None
        self.pixsize = None
        self.angle = None
        self.q = np.array([1., 0., 0., 0.], 'float')
        self.t_init = None
        self.t = None
        self.xyz = None
        self.xyz_sum = None
        self.vol = None
        self.bin_idx = None
        self.binfsc = None
        self.afsc = None
        self.nbin = None
        self.ibin = None
        self.fobj = None

    def calc_fsc(self):
        assert self.e0.shape == self.e1.shape == self.bin_idx.shape
        binfsc, _, bincounts = fsctools.anytwomaps_fsc_covariance(
            self.e0, self.e1, self.bin_idx, self.nbin)
        self.binfsc = binfsc
        # weighted average
        vals, counts = binfsc[:self.ibin], bincounts[:self.ibin]
        self.afsc = np.sum(vals * counts) / np.sum(counts)

    def get_wght(self): 
        nz, ny, nx = self.e0.shape
        val_arr = np.zeros((self.nbin, 2), dtype='float')
        val_arr[:,0] = self.binfsc #/ (1 - self.binfsc ** 2)
        self.wgrid = fc.read_into_grid2(self.bin_idx,
            val_arr, self.nbin, nz, ny, nx)[:,:,:,0]

    def numeric_deri(self, x, info):
        # 1st derivative from finite difference central formula
        wgrid = 1.0
        nx, ny, nz = self.e0.shape
        df = np.zeros(5, dtype="float") # Jacobian
        tpi = (2.0 * np.pi * 1j)
        for i in range(3):
            df[i] = -np.sum(np.real(wgrid * np.conjugate(self.e0) * (self.e1 * tpi * self.sv[i,:,:,:])))
        dphi = dthet = 1/(2*nx)
        rm_f1, _, _ = rotmat_spherical_crd(phi=self.phi+dphi, thet=self.thet, angle=self.angle)
        rm_b1, _, _ = rotmat_spherical_crd(phi=self.phi-dphi, thet=self.thet, angle=self.angle)
        rm_f2, _, _ = rotmat_spherical_crd(phi=self.phi, thet=self.thet+dthet, angle=self.angle)
        rm_b2, _, _ = rotmat_spherical_crd(phi=self.phi, thet=self.thet-dthet, angle=self.angle)
        t1 = timer()
        dfrs = fc.numberic_derivatives(
            self.e0, 
            self.bin_idx,
            np.stack([rm_f1, rm_b1, rm_f2, rm_b2], axis=0),
            0,
            self.ibin,
            4,
            nx,ny,nz
            )
        #print('time for interpolation total: ', timer()-t1)
        dFdphi = self.st * dfrs[0,:,:,:] / (2*dphi)
        df[3] = -np.sum(wgrid * np.real(self.e0 * np.conjugate(dFdphi)))
        dFdthet = self.st * dfrs[1,:,:,:] / (2*dthet)
        df[4] = -np.sum(wgrid * np.real(self.e0 * np.conjugate(dFdthet)))
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
        nx, ny, nz = self.e0.shape
        t = np.asarray(self.t_init, 'float') + x[:3]
        self.st = fc.get_st(nx, ny, nz, t)[0]
        dphi, dthet = x[3], x[4]
        self.phi = self.phi0 + dphi
        self.thet = self.thet0 + dthet
        self.rotmat, _, _ = rotmat_spherical_crd(
            phi=self.phi, thet=self.thet, angle=self.angle)
        ax = spherical2ax(phi=self.phi, thet=self.thet)
        self.e1 = self.st * rotate_f(self.rotmat, self.e0, self.bin_idx, self.ibin)[:, :, :, 0]
        self.calc_fsc()
        self.get_wght()
        fval = np.sum(self.wgrid * self.e0 * np.conjugate(self.e1))
        t2 = np.asarray([nx*t[0], ny*t[1], nz*t[2]], 'float') * np.asarray(self.pixsize)
        print('cycle, fval, afsc, ax, trans ', info['Nfeval'], fval.real, self.afsc, ax, t2)
        self.fobj.write('cycle=%i, fval=%.3f, afsc=%.3f, ax=%s, trans=%s\n' %(
            info['Nfeval'], 
            fval.real, 
            self.afsc, 
            vec2string(ax), 
            vec2string(t2), 
            ))
        info['Nfeval'] += 1
        return -fval.real

    def optimize(self):
        from scipy.optimize import minimize
        self.xyz = create_xyz_grid(self.e0.shape)
        grid2 = create_xyz_grid2(self.e0.shape)
        self.sv = np.stack(grid2, axis = 0)
        nx, ny, nz = self.e0.shape
        t = np.array([0., 0., 0.], 'float')
        self.st = fc.get_st(nx, ny, nz, t)[0]
        tol = 1e-4
        self.phi = self.phi0
        self.thet = self.thet0
        print('initial values of phi, theta, angle:')
        print(self.phi0, self.thet0, self.angle)
        x = np.array([0., 0., 0., 0., 0.], 'float')
        options = {'maxiter':2000}
        args=({'Nfeval':0},)
        #self.method = 'L-BFGS-B'
        print("Optimization method: ", self.method)
        self.fobj.write("Optimization method: %s\n" %self.method)
        result = minimize(
            fun=self.functional, 
            x0=x, 
            method=self.method,
            jac=self.numeric_deri,
            #hess=self.secondd,
            tol=tol,
            options=options,  
            args=args,
            )
        self.fobj.write('\n')
        if result.status:
            print(result)
            self.fobj.write('%s' % result)
        self.fobj.write('\n')
        self.t = np.asarray(self.t_init, 'float') + result.x[:3]
        self.phi = self.phi0 + result.x[3]
        self.thet = self.thet0 + result.x[4]        

def fsc_between_static_and_transfomed_map(emmap1, rm, t, ergi=None, ibin=None):
    print('RM for FSC calculation')
    print(rm)
    print('t: ', t)
    t = -np.asarray(t, 'float')
    fo=emmap1.fo_lst[0]
    eo = fo
    #eo=emmap1.eo_lst[0]
    bin_idx=emmap1.bin_idx
    nbin=emmap1.nbin
    if ibin is None: ibin = nbin
    nx, ny, nz = eo.shape
    st, _, _, _ = fc.get_st(nx, ny, nz, t)
    if ergi is not None:
        ert = st * get_f(eo, ergi[0], ergi[1], rm)
    else:
        # test - 1st translate then rotate
        map2 = (ifftshift((ifftn(ifftshift(st * eo))).real))
        # rotate in real space
        map2 = fc.trilinear_map(rm, map2, 0, nx, ny, nz)
    ert = fftshift(fftn(fftshift(map2)))
    f1f2_fsc = fsctools.anytwomaps_fsc_covariance(
        eo, ert, bin_idx, nbin)[0]
    return f1f2_fsc, ert


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
    initial_axis  = spherical2ax(phi, thet)
    initial_t = t
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
                # output FSC between static and initial sym. copy
                emdalogger.log_string(fobj, 'FSC between static and initial copy')
                emdalogger.log_fsc(
                    fobj,
                    {
                        'Res.':emmap1.res_arr,
                        'FSC':f1f2_fsc
                    }
                )
                emdalogger.log_newline(fobj)
                ang = float(180/np.pi * angle)
                if emmap1.output_maps:
                    output_mapname = emmap1.emdbid+'_rotated_ax_'+combinestring(initial_axis)+"_ang"+str(round(ang,2))+".mrc"
                    emdalogger.log_string(fobj, 'writing out %s'%output_mapname)
                    transformedmap = np.real(ifftshift(ifftn(ifftshift(frt))))
                    tm1 = iotools.Map(output_mapname)
                    tm1.arr = transformedmap
                    tm1.cell = emmap1.map_unit_cell
                    tm1.origin = [0,0,0]
                    tm1.write()
                fsc_lst.append(f1f2_fsc)
                if emmap1.fitfsc > 0.999:
                    emdalogger.log_string(
                        fobj, "\n***FSC between static and moving maps***")
                    emdalogger.log_fsc(
                        fobj,
                        {
                            'Resol.':res_arr,
                            'StartFSC':fsc_lst[0],
                            'EndFSC':fsc_lst[0]
                        }
                    )
                    break
                ibin = get_ibin(filter_fsc(f1f2_fsc), cutoff=emmap1.fitfsc)
                ibin_old = ibin
                if ibin >= 5:
                    if fitbin <= ibin:
                        ibin = fitbin
                    ibin_old = ibin
                    emdalogger.log_string(fobj, "Fitting starts at %s (A)" %(emmap1.res_arr[ibin]))
                else:
                    print('Resolution= %s (A)' %emmap1.res_arr[ibin])
                    print('FSC between two copies is too low. FSC= %s ibin=%s' %(f1f2_fsc[ibin], ibin))
            else:
                f1f2_fsc, frt = fsc_between_static_and_transfomed_map(
                    emmap1=emmap1,
                    rm=rotmat,
                    t=t,
                )          
                ibin = get_ibin(filter_fsc(f1f2_fsc), cutoff=emmap1.fitfsc)
                if fitbin < ibin:
                    ibin = fitbin
                if ibin_old == ibin:
                    print('ibin_old, ibin: ')
                    print(ibin_old, ibin)
                    fsc_lst.append(f1f2_fsc)
                    res_arr = emmap1.res_arr[:ibin]
                    fsc_bef = fsc_lst[0][:ibin]
                    fsc_aft = fsc_lst[1][:ibin]
                    refined_fsc = fsc_lst[1]
                    emdalogger.log_string(
                        fobj, "\n***FSC between static and moving maps***")
                    emdalogger.log_fsc(
                        fobj,
                        {'Resol.':res_arr,
                         'StartFSC':fsc_bef,
                         'EndFSC':fsc_aft}
                    )
                    plotname = emmap1.emdbid+'_fsc_ax'+combinestring(initial_axis)+"_ang"+str(round(ang,2))
                    emdalogger.log_newline(fobj)
                    emdalogger.log_string(
                        fobj,
                        'Plotting FSCs to %s'%plotname
                    )
                    # plotting Proshade axis, EMDA axis FSCs with fullmap FSC
                    plotter.plot_nlines(
                        res_arr=emmap1.res_arr, 
                        list_arr=[fsc_lst[0], fsc_lst[1], emmap1.fscfull],
                        curve_label=["Proshade axis", "EMDA axis", "Fullmap FSC"], 
                        plot_title="FSC based on Symmetry axis", 
                        fscline=1.,
                        mapname=plotname,
                        linecolor=['red', 'green', 'black'],
                        verticleline=emmap1.claimed_bin,
                        multicolor=True,
                        )
                    if emmap1.output_maps:                    
                        emdalogger.log_string(
                            fobj, 'Outputting static_map.mrc'
                        )
                        stmap = np.real(ifftshift(ifftn(ifftshift(emmap1.fo_lst[0]))))
                        #stmap = np.real(ifftshift(ifftn(ifftshift(emmap1.eo_lst[0]))))
                        stm = iotools.Map('static_map.mrc')
                        stm.arr = stmap
                        stm.cell = emmap1.map_unit_cell
                        stm.origin = [0,0,0]
                        stm.write()
                        fittedmapname = emmap1.emdbid+'_fitted_ax'+combinestring(initial_axis)+"_ang"+str(round(ang,2))+".mrc"
                        emdalogger.log_string(
                            fobj, 'Outputting %s'%fittedmapname
                        )
                        transformedmap = np.real(ifftshift(ifftn(ifftshift(frt))))
                        tm = iotools.Map(fittedmapname)
                        tm.arr = transformedmap
                        tm.cell = emmap1.map_unit_cell
                        tm.origin = [0,0,0]
                        tm.write()     
                    break
                elif ibin_old > ibin:
                    is_abandon = True
                    emdalogger.log_string(
                        fobj, 
                        'axis refinement went wrong! ibin_old=%i, ibin=%i' %(ibin_old, ibin)
                    )
                    final_axis = initial_axis
                    final_t = initial_t
                    pos_ax = []
                    refined_fsc = fsc_lst[0]
                    break
                else:
                    ibin_old = ibin
                    emdalogger.log_string(
                        fobj, "Fitting starts at %s (A)" %(emmap1.res_arr[ibin]))
            if ibin >= 5:
                e_list = [emmap1.fo_lst[0], emmap1.eo_lst[0]]
                fcut, cBIdx, cbin = cut_resolution_for_linefit(
                    e_list, emmap1.bin_idx, emmap1.res_arr, ibin
                )
                bfgs = Bfgs()
                bfgs.fobj = fobj
                bfgs.phi0 = phi
                bfgs.thet0 = thet
                bfgs.angle = angle
                bfgs.t_init = t
                bfgs.e0 = fcut[1, :, :, :] #static_cutmap
                bfgs.bin_idx = cBIdx #emmap1.bin_idx 
                bfgs.nbin = cbin #emmap1.nbin#cbin
                bfgs.ibin = ibin
                bfgs.pixsize = emmap1.pix
                bfgs.method = optmethod
                bfgs.optimize()
                t = bfgs.t
                phi = bfgs.phi
                thet = bfgs.thet
                rotmat, _, _ = rotmat_spherical_crd(phi, thet, angle)
                final_axis = spherical2ax(phi, thet)
                final_t = t
            else:
                is_abandon = True
                emdalogger.log_string(
                    fobj,
                    "ibin = %s, Cannot proceed axis refinement." %(ibin)
                )
                final_axis = initial_axis
                final_t = initial_t
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
    optmethod='L-BFGS-B',
    **kwargs
):
    #frt = kwargs['frt']
    if fobj is None:
        fobj = open('EMDA_axis-refinement.txt', 'w')
    axis = np.asarray(rotaxis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    emdalogger.log_newline(fobj)
    emdalogger.log_string(
        fobj, "Number of refinement cycles: %i"%emmap1.ncycles
    )
    emdalogger.log_string(
        fobj, "Data resolution for refinement: % .2f"%emmap1.fitres
    )
    angle = np.deg2rad(float(360.0 / symorder))
    phi, theta = ax2spherical(axis)
    args = {'phi':phi, 
            'thet':theta, 
            'angle':angle, 
            't_init':np.asarray(t_init, dtype="float"),
            'optmethod':optmethod,
            #'frt':frt
            }
    results = run_fit(
        emmap1=emmap1,
        fobj=fobj,
        args=args,
    )
    return results


def singleaxisrefine(imap, imask, symorder, axis):
    m1 = iotools.Map(imap)
    m1.read()
    mm = iotools.Map(imask)
    mm.read()
    emmap1 = EmmapOverlay(arr=m1.workarr)
    emmap1.map_unit_cell = m1.workcell
    emmap1.prep_data()
    emmap1.mask = mm.workarr
    emmap1.ncycles = 10
    emmap1.fitres = 5.0
    emmap1.fitfsc = 0.1
    emmap1.eo_lst = emmap1.fo_lst
    results = axis_refine(
        emmap1=emmap1,
        rotaxis=axis,
        symorder=symorder, 
        )

    print(results)


def prepare_data_using_halfmaps(half1, imask, axis, symorder, resol=None, fobj=None):
    import re
    from emda2.ext import mapmask
    import emda2.emda_methods2 as em
    from symanalysis_pipeline import writemap
    from emda2.core import maptools
    import fcodes2
    if resol is not None:
        resol = resol * 1.1 # taking 10% less resolution of author claimed
    # open halfmaps
    half2 = half1.replace("map_1", "map_2")
    print(half1)
    print(half2)
    h1 = iotools.Map(half1)
    h1.read()
    h2 = iotools.Map(half2)
    h2.read()    

    m = re.search('emd_(.+)_half', half1)
    logname = 'emd-%s-pointgroup.txt'%m.group(1)
    emdbid = 'emd-%s'%m.group(1)
    if fobj is None:
        fobj = open(logname, 'w')
    maskname = 'emda_mapmask_emd-'+m.group(1)+'.mrc'
    reboxedmaskname = 'emda_rbxmapmask_emd-'+m.group(1)+'.mrc'
    reboxedmapname = 'emda_rbxfullmap_emd-'+m.group(1)+'.mrc'

    # get the mask for future calculations
    if imask is None:
        # calculate EMDA mask from map
        mapmask.main(imap=half1, imask=maskname)
        mm = iotools.Map(name=maskname)
        mm.read()
    else:
        mm = iotools.Map(name=imask)
        mm.read()
        
    # reboxing halfmaps using the mask
    print('Reboxing...')
    #padwidth = int(10/h1.workcell[0]/h1.workarr.shape[0])
    padwidth = 10
    rmap1, rmask = em.rebox_by_mask(arr=h1.workarr, mask=mm.workarr, mask_origin=mm.origin, padwidth=padwidth)
    rmap2, rmask = em.rebox_by_mask(arr=h2.workarr, mask=mm.workarr, mask_origin=mm.origin, padwidth=padwidth)
    fullmap = (rmap1 + rmap2) / 2
    # write out reboxed fullmap and mask
    newcell = [fullmap.shape[i]*h1.workcell[i]/shp for i, shp in enumerate(h1.workarr.shape)]
    for _ in range(3): newcell.append(90.0)
    writemap(fullmap, newcell, reboxedmapname)
    writemap(rmask, newcell, reboxedmaskname)
    # create resolution grid
    nbin, res_arr, bin_idx, sgrid = em.get_binidx(cell=newcell, arr=rmap1)

    claimed_res = float(resol)
    dist = np.sqrt((res_arr - claimed_res) ** 2)
    claimed_cbin = np.argmin(dist)
    if res_arr[claimed_cbin] <= claimed_res:
        claimed_cbin -= 1
    claimed_res = res_arr[claimed_cbin]
    print('Claimed resolution and cbin: ', claimed_res, claimed_cbin)

    binfsc = em.fsc(
        f1=fftshift(fftn(rmap1 * rmask)), 
        f2=fftshift(fftn(rmap2 * rmask)), 
        bin_idx=bin_idx, 
        nbin=nbin
        )
    fsc_full = 2 * binfsc / (1. + binfsc)
    fo = fftshift(fftn(fftshift(fullmap * rmask)))
    nx, ny, nz = fo.shape
    eo = fcodes2.get_normalized_sf_singlemap(
        fo=fo,
        bin_idx=bin_idx,
        nbin=nbin,
        mode=0,
        nx=nx,ny=ny,nz=nz,
        )
    fsc_full = filter_fsc(fsc_full, thresh=0.1)
    fsc_star = np.sqrt(fsc_full)
    eo = eo * fcodes2.read_into_grid(bin_idx, fsc_star, nbin, nx, ny, nz)
    
    emmap1 = EmmapOverlay(arr=fullmap)
    emmap1.map_unit_cell = newcell
    emmap1.prep_data()
    emmap1.mask = rmask
    emmap1.ncycles = 10
    emmap1.fitres = 5.0
    emmap1.fitfsc = 0.1
    emmap1.fo_lst = [fo]
    emmap1.eo_lst = [eo]
    emmap1.output_maps = True
    emmap1.emdbid = emdbid
    emmap1.fscfull = fsc_full
    emmap1.claimed_res = claimed_res
    emmap1.claimed_bin = claimed_cbin
    results = axis_refine(
        emmap1=emmap1,
        rotaxis=axis,
        symorder=symorder, 
        )
    print(results)



if __name__ == "__main__":
    #imap = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-0608/emd_0608.map"
    #imap =  "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-0139/emda_reboxedmap_emd-0139.map.mrc"
    #imap = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-0689/emd_0689.map"
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

    #imap = "emda_rbxfullmap_emd-12819.mrc"
    #imask = "emda_rbxmapmask_emd-12819.mrc"
    #symorder = 5
    #axis = [0.5293043277784694, 0.0035496841491803976, 0.8484246155890478]
    #singleaxisrefine(imap, imask, symorder, axis)


    #half1="/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12819/emd_12819_half_map_1.map"
    #imask="/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12819/manual_emda_mask_0.mrc"
    #axis=[0.5237267504834436, -0.010016429961688136, 0.8518274249863498]
    #symorder=5
    half1 = "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-21231/emd_21231_half_map_1.map"
    imask = "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-21231/emd_21231_msk_1.map"
    axis = [0.002,  0.020,  1.000]
    symorder = 3
    resol = 4.6
    prepare_data_using_halfmaps(
        half1=half1, 
        imask=imask, 
        axis=axis, 
        symorder=symorder,
        resol=resol)