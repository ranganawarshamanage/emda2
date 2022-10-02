from emda2.ext.sym import axis_refinement_new2
#from axis_refinement_new2 import axis_refine
from emda2.core import iotools
import emda2.ext.utils as utils
from numpy.fft import fftn, ifftn, fftshift
import fcodes2
import numpy as np
import emda2.emda_methods2 as em


def get_rotation_center(m1, axis, order, claimed_res, mm):
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
    # using COM
    nx, ny, nz = map1.shape
    com = utils.center_of_mass_density(map1)
    #fobj.write('Centre of Mass [x, y, z] (pixel units) %s\n' %list(com))
    print("com:", com)
    box_centr = (nx // 2, ny // 2, nz // 2)
    map1_com_adjusted = utils.shift_density(map1, np.subtract(box_centr, com))
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
    emmap1 = axis_refinement_new2.EmmapOverlay(arr=map1_com_adjusted)
    emmap1.claimedres_data = claimedres_data
    emmap1.bin_idx = bin_idx
    emmap1.mask = mm.workarr
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
    results = axis_refinement_new2.axis_refine(
        emmap1=emmap1,
        rotaxis=axis,
        symorder=order,
        optmethod='bfgs'
    )
    if len(results) > 0:
        final_axis, final_t, pos_ax, refined_fsc = results
        print('Rotation Centre [x, y, z] (A) %s\n' %[pos_ax[i] for i in range(3)])
    return results


def main(imap, axis, order, imask, resol):
    m1 = iotools.Map(imap)
    m1.read()
    mm = iotools.Map(imask)
    mm.read()
    results = get_rotation_center(m1, axis, order, resol, mm)



if __name__=="__main__":
    imap = "/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3651/other/rotation_centre/emda_rbxfullmap_emd-3651.mrc"
    imask = "/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3651/other/rotation_centre/emda_rbxmapmask_emd-3651.mrc"
    axis = [2.94523737e-03, 2.89148106e-06, 9.99995663e-01]
    order = 2
    resol = 4.
    main(
        imap=imap,
        axis=axis,
        order=order,
        resol=resol,
        imask=imask,
    )