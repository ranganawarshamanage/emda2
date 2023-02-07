"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

import math
from emda2.ext.sym import axis_refinement
from emda2.core import iotools, fsctools, quaternions
import emda2.ext.utils as utils
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import fcodes2
import numpy as np
import emda2.emda_methods2 as em
from timeit import default_timer as timer
from emda2.ext.overlay import run_fit


def get_rotation_center(m1, axis, order, claimed_res, mm):
    results = []
    nbin, res_arr, bin_idx, sgrid = em.get_binidx(cell=m1.workcell, arr=m1.workarr)
    if claimed_res < 5:
        resol4refinement = float(5)
    else:
        resol4refinement = claimed_res
    # select data upto claimed resol
    _, map1 = utils.lowpassmap_butterworth(
        fc=m1.workarr, 
        sgrid=sgrid,
        smax=claimed_res*1.1, 
        order=4)
    map1 = map1 * mm.workarr
    # using COM
    nx, ny, nz = map1.shape
    com = utils.center_of_mass_density(map1)
    #fobj.write('Centre of Mass [x, y, z] (pixel units) %s\n' %list(com))
    print("com:", com)
    box_centr = (nx // 2, ny // 2, nz // 2)
    map1_com_adjusted = utils.shift_density(map1, np.subtract(box_centr, com))
    mask_com_adjusted = utils.shift_density(mm.workarr, np.subtract(box_centr, com))
    #map1_com_adjusted = map1
    #mask_com_adjusted = mm.workarr
    com = utils.center_of_mass_density(map1_com_adjusted)
    fo = fftshift(fftn(fftshift(map1_com_adjusted)))
    eo = fcodes2.get_normalized_sf_singlemap(
        fo=fo,
        bin_idx=bin_idx,
        nbin=nbin,
        mode=0,
        nx=nx,ny=ny,nz=nz,
        )
    #fsc_max = np.sqrt(filter_fsc(fsc_full))
    #eo = eo * fcodes2.read_into_grid(bin_idx, fsc_max, nbin, nx, ny, nz)
    dist = np.sqrt((res_arr - claimed_res) ** 2)
    claimed_cbin = np.argmin(dist) + 1
    claimedres_data = [fo, bin_idx, claimed_res, nbin, res_arr, claimed_cbin]  
    tlist = [0., 0., 0.]
    emmap1 = axis_refinement.EmmapOverlay(arr=map1_com_adjusted)
    emmap1.claimedres_data = claimedres_data
    emmap1.bin_idx = bin_idx
    emmap1.mask = mask_com_adjusted#mm.workarr
    emmap1.res_arr = res_arr
    emmap1.nbin = nbin
    emmap1.pix = [m1.workcell[i]/sh for i, sh in enumerate(m1.workarr.shape)]
    emmap1.fo_lst = [fo]
    emmap1.eo_lst = [eo]
    emmap1.fitres = resol4refinement
    emmap1.fitfsc = 0.15
    emmap1.ncycles = 10
    emmap1.symdat = []
    emmap1.com = True
    emmap1.com1 = com
    emmap1.map_dim = eo.shape
    emmap1.map_unit_cell = m1.workcell
    emmap1.pix = [m1.workcell[i]/sh for i, sh in enumerate(m1.workarr.shape)]
    # refine axis and translation
    """ results = axis_refinement_new2.axis_refine(
        emmap1=emmap1,
        rotaxis=axis,
        symorder=order,
        optmethod='L-BFGS-B'
    )
    if len(results) > 0:
        final_axis, final_t, pos_ax, refined_fsc = results
        print('Rotation Centre [x, y, z] (A) %s\n' %[pos_ax[i] for i in range(3)])
        print('final axis: ', final_axis)
        average_translation(emmap1, final_axis, order) """

    tlist = refine_translation(emmap1, axis, order)
    rotcentre = get_centroid_coord(com, tlist, emmap1.map_dim)
    return emmap1, rotcentre


def get_t_to_centroid(emmap1, axis, order):
    tlist = refine_translation(emmap1, axis, order)
    rotcentre = get_centroid_coord(emmap1.com1, tlist, emmap1.map_dim)
    temp_t = np.subtract(rotcentre, emmap1.com1)
    t_to_centroid = [temp_t[i]/emmap1.map_dim[i] for i in range(3)]
    return t_to_centroid


class Opt_trans:
    def __init__(self):
        self.e0 = None
        self.e1 = None
        self.ert = self.e1
        self.ereal_rgi = None
        self.eimag_rgi = None
        self.wgrid = None
        self.sv = None
        self.q = None
        self.t = None
        self.bin_idx = None
        self.nbin = None
        self.binfsc = None

    def hess_r(self, x):
        st1 = timer()
        tp2 = (2.0 * np.pi) ** 2
        ddf = np.zeros((3, 3), dtype="float")
        for i in range(3):
            for j in range(3):
                if i == 0 or (i > 0 and j >= i):
                    #ddf[i,j] = -tp2 * np.sum(self.wgrid * self.sv[i] * self.sv[j]) #/ self.vol
                    ddf[i,j] = -tp2 * np.sum(self.wgrid * self.sv[i]**2) / 3
                else:
                    ddf[i, j] = ddf[j, i]
        et1 = timer()
        print("Hess time: ", et1-st1)
        return -ddf
        
    def derivatives(self, x):
        st1 = timer()
        # translation derivatives
        df = np.zeros(3, dtype="float")
        for i in range(3):
            df[i] = np.sum(np.real(self.e0_conj * (self.ert * self.svtpi[i])))
        et1 = timer()
        print("deri time: ", et1-st1)
        return -df

    def calc_fsc(self):
        st1 = timer()
        assert self.e0.shape == self.e1.shape == self.bin_idx.shape
        binfsc, _, bincounts = fsctools.anytwomaps_fsc_covariance(
            self.e0, self.ert, self.bin_idx, self.nbin)
        self.binfsc = binfsc
        #self.avgfsc = ut.get_avg_fsc(binfsc=binfsc, bincounts=bincounts)
        self.avgfsc = np.average(binfsc)
        et1 = timer()
        print('fsc calc time: ', et1-st1)

    def get_wght(self): 
        st1 =  timer()
        nz, ny, nx = self.e0.shape
        val_arr = np.zeros((self.nbin, 2), dtype='float')
        val_arr[:,0] = self.binfsc #/ (1 - self.binfsc ** 2)
        self.wgrid = fcodes2.read_into_grid2(self.bin_idx,
            val_arr, self.nbin, nz, ny, nx)[:,:,:,0]
        et1 = timer()
        print("wegiht calc time: ", et1-st1)

    #3. functional
    def functional(self, x):
        nx, ny, nz = self.e1.shape
        self.t = x
        st, sv1, sv2, sv3 = fcodes2.get_st(nx, ny, nz, self.t)
        self.sv = [sv1, sv2, sv3]
        self.ert = self.e1 * st
        self.calc_fsc()
        self.get_wght()
        st1 = timer()
        fval = np.sum(self.wgrid * self.e0_conj * self.ert)
        print('fval, avgfsc, t: ', fval.real, self.avgfsc, self.t)
        et1 = timer()
        print('func time: ', et1-st1)
        return -fval.real

    #3. optimize using BFGS
    def optimize(self):
        from scipy.optimize import minimize
        #precalculated for speed
        self.e0_conj = np.conjugate(self.e0)
        self.tpi = (2.0 * np.pi * 1j)
        nx, ny, nz = self.e0.shape
        _, sv1, sv2, sv3 = fcodes2.get_st(nx, ny, nz, [0., 0., 0.])
        self.svtpi = [sv1*self.tpi, sv2*self.tpi, sv3*self.tpi]
        self.vol = nx * ny * nz
        x = np.array([0., 0., 0.], 'float')
        options = {'maxiter': 100}
        #result = minimize(fun=self.functional, x0=x, method='Newton-CG', jac=self.derivatives, hess=self.hess_r, tol=1e-5) # worked
        result = minimize(fun=self.functional, x0=x, method='L-BFGS-B', jac=self.derivatives, tol=1e-5, options=options) # worked
        print(result)
        self.t = result.x 
        print('translation vec: ', self.t) 


def refine_translation(emmap1, axis, order):
    from emda2.core import maptools
    from emda2.ext.sym.axis_refinement import rotate_f
    # rotate the map by different angles and optimize the translation
    # then average to get the translation

    eo = emmap1.eo_lst[0]
    fo = emmap1.fo_lst[0]
    t = np.array([0., 0., 0.], 'float')
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    print('')
    print('axis: ', axis)
    print('******* Translation only refinement *******')
    tlist = []
    for i in range(order-1):
        j = i + 1
        angle = float(360 * j/order)
        print('angle: ', angle)
        q = quaternions.get_quaternion(list(axis), angle)
        #rotmat = quaternions.get_RM(q)
        #f_transformed = maptools.transform_f(
        #                flist=[fo, eo],
        #                axis=axis,
        #                translation=t,
        #                angle=angle
        #                )
        f_transformed = rotate_f(
            rm=quaternions.get_RM(q), 
            f=np.stack([fo, eo], axis=-1), 
            bin_idx=emmap1.bin_idx, 
            ibin=emmap1.nbin)
        emmap1.fo_lst = [fo, f_transformed[:,:,:,0]]
        emmap1.eo_lst = [eo, f_transformed[:,:,:,1]]
        # this is using emda overlay optimisation for translation
        #emmap1.fo_lst = [fo, f_transformed[0]]
        #emmap1.eo_lst = [eo, f_transformed[1]]
        emmap1.map_origin = [0,0,0]
        emmap1.comlist = []
        emmap1.pixsize = emmap1.pix
        results = run_fit(
            emmap1=emmap1,
            rotmat=np.identity(3),
            t=np.array([0., 0., 0.], 'float'),
            ncycles=10,
            ifit=1,
            fitres=emmap1.fitres,
            t_only=True,
            )
        if results is not None:
            t_refined, q_final = results
            print('t_refined:', t_refined)
            tlist.append(t_refined)

        # this is scipy.optimize method and using 2nd derivative (analytical)
        """ Pt = Opt_trans()
        Pt.e0 = eo
        Pt.e1 = f_transformed[1]
        Pt.bin_idx = emmap1.bin_idx
        Pt.nbin = emmap1.nbin
        Pt.q = q
        Pt.optimize()
        t_refined = Pt.t
        tlist.append(t_refined) """
        # recalculate FSC now for comparison.
        #map2 = (ifftshift((ifftn(ifftshift(f_transformed[0]))).real)) * emmap1.mask
        #frt = fftshift(fftn(fftshift(map2)))
        #binfsc0, _, bincounts = fsctools.anytwomaps_fsc_covariance(
        #    fo, frt, emmap1.bin_idx, emmap1.nbin)
        #nx, ny, nz = eo.shape
        #f1 = frt * fcodes2.get_st(nx, ny, nz, t_refined)[0]
        #binfsc1, _, bincounts = fsctools.anytwomaps_fsc_covariance(
        #    fo, f1, emmap1.bin_idx, emmap1.nbin)
        #for i in range(emmap1.nbin):
        #    print(i, emmap1.res_arr[i], binfsc0[i], binfsc1[i])
    return tlist


def get_centroid_coord(com, tlist, shape):
    """ 
    Returns the centroid of the polygon, which is the
    same as the centre of the rotation
    com - center of the mass of the initial map
    tlist - list of translations (each t is the refined
        translation)
    shape - shape of the initial map
    """
    print('shape: ', shape)
    print('com: ', com)
    print('tlist: ', tlist)
    nsym = len(tlist) + 1
    nx, ny, nz = shape
    px, py, pz = list(com)
    for t in tlist:
        px += com[0]+t[0]*nx
        py += com[1]+t[1]*ny
        pz += com[2]+t[2]*nz
    return (px/nsym, py/nsym, pz/nsym)


def main(imap, axis, order, imask, resol):
    m1 = iotools.Map(imap)
    m1.read()
    mm = iotools.Map(imask)
    mm.read()
    emmap1, rotcentre = get_rotation_center(m1, axis, order, resol, mm)
    print('rotation centre: ', rotcentre)
    temp_t = np.subtract(rotcentre, emmap1.com1)
    t_to_centroid = [temp_t[i]/emmap1.map_dim[i] for i in range(3)]
    return rotcentre, t_to_centroid




if __name__=="__main__":
    #imap = "/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3651/other/rotation_centre/emda_rbxfullmap_emd-3651.mrc"
    #imask = "/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3651/other/rotation_centre/emda_rbxmapmask_emd-3651.mrc"
    #imap = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-10561/EMD-10561/emda_rbxfullmap_emd-10561.mrc"
    #imask = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-10561/EMD-10561/emda_rbxmapmask_emd-10561.mrc"
    imap = "/Users/ranganaw/MRC/REFMAC/EMD-6952/map/test/translated_1A_010.mrc"
    imask = "/Users/ranganaw/MRC/REFMAC/EMD-6952/map/test/manual_emda_mask_0.mrc"
    #axis = [2.94523737e-03, 2.89148106e-06, 9.99995663e-01] # order=2 # emd3651
    axis = [0.0, 0.0, 1.0]
    order = 3
    resol = 4.5
    main(
        imap=imap,
        axis=axis,
        order=order,
        resol=resol,
        imask=imask,
    )