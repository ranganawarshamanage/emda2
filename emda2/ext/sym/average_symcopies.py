import numpy as np
import fcodes2
from emda2.core import quaternions, fsctools, restools, iotools, plotter
import math
from emda2.ext.utils import rotate_f
import emda2.emda_methods2 as em
from emda2.ext.utils import (
    shift_density, 
    center_of_mass_density)
from emda2.ext.sym import get_rotation_center as grc


def transform(fo, axis, angle):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    q = quaternions.get_quaternion(list(axis), angle)
    rotmat = quaternions.get_RM(q)
    return rotate_f(rotmat, fo, interp="linear")[:, :, :, 0]


def average(fo, axis, fold, t):
    nx, ny, nz = fo.shape
    st = fcodes2.get_st(nx, ny, nz, t)[0]
    fo = fo*st
    f_sum = fo
    anglist = [float(360*i/fold) for i in range(1, fold)]
    for angle in anglist:
        f_sum += transform(fo, axis, angle)
    return f_sum/fold

def average2(fo, axes, folds, tlist):
    nx, ny, nz = fo.shape
    i = 1
    for axis, t, fold in zip(axes, tlist, folds):
        t = -np.asarray(t, 'float') # reverse the translation. 
        st = fcodes2.get_st(nx, ny, nz, t)[0]
        fo = fo*st
        if i == 1: f_sum = fo
        anglist = [float(360*i/fold) for i in range(1, fold)]
        for angle in anglist:
            f_sum += transform(fo, axis, angle)
            i += 1
    return f_sum/i

def main(f_list, axes, folds, tlist, **kwargs):
    fhf1_avg = average2(fo=f_list[0], axes=axes, tlist=tlist, folds=folds)
    fhf2_avg = average2(fo=f_list[1], axes=axes, tlist=tlist, folds=folds)
    #fhf1_avg = average(fo=f_list[0], axis=axes[0], fold=folds[0], t=tlist[0])    
    #fhf2_avg = average(fo=f_list[1], axis=axes[0], fold=folds[0], t=tlist[0])
    # compute FSC
    bin_idx = kwargs['bin_idx']
    nbin = kwargs['nbin']

    binfsc1 = fsctools.anytwomaps_fsc_covariance(
            f_list[0], f_list[1], bin_idx, nbin)[0]
    binfsc2 = fsctools.anytwomaps_fsc_covariance(
            fhf1_avg, fhf2_avg, bin_idx, nbin)[0]
        
    return binfsc1, binfsc2#, fhf1_avg




if __name__ == "__main__":

    #12139
    #half1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-12139/reboxed_emd_12139_half_map_1.mrc"
    #mask = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-12139/emda_reboxedmask.mrc"
    #axes = [
    #    [0., 0., 1.], # 3-fold
    #    [1., 0., 0.], # 2-fold
    #    ] 
    ##t = [-1.29307488e-03,  8.91764397e-04,  1.59117856e-07] # 3-fold
    ##axis = [1., 0., 0.] # 2-fold
    ##t = [5.26008350e-08/2, -5.82664647e-06/2, -6.17546617e-06/2] # 2-fold
    #folds = [3, 2]
    #resol = 3.1

    #23884
    #half1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-23884/emd_23884_half_map_1.map"
    #mask = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-23884/emd_23884_msk_1.map"
    #axis = [0., 0., 1.]
    #t = [-3.27695196e-04/2,  4.42089217e-04/2, -5.88523441e-06/2]
    #fold = 2
    #resol = 3.8

    # 30217
    #half1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-30217/emd_30217_half_map_1.map"
    #mask = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-30217/emd_30217_msk_1.map"
    #axes = [
    #    [0.,0.,1.]
    #]
    #folds = [2]
    #resol = 2.8

    # 13803
    #half1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-13803/emd_13803_half_map_1.map"
    #mask = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-13803/emd_13803_msk_1.map"
    #axes = [
    #    [0., 0., 1.]
    #]
    #folds  = [2]
    #resol = 3.78

    # 10561
    half1 = "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10561/emd_10561_half_map_1.map"
    mask = "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10561/emda_mapmask.mrc"
    #ax4:  0.000  0.000  1.000 Order: 7
    axes = [
        [0.000,  0.000,  1.000]
    ]
    folds = [7]
    resol = 4.5

    half2 = half1.replace("map_1", "map_2")
    h1 = iotools.Map(half1)
    h1.read()
    h2 = iotools.Map(half2)
    h2.read()
    mm = iotools.Map(mask)
    mm.read()

    rmap1, rmask = em.rebox_by_mask(arr=h1.workarr, mask=mm.workarr)
    rmap2, rmask = em.rebox_by_mask(arr=h2.workarr, mask=mm.workarr)

    fullmap = (rmap1 + rmap2) / 2

    newcell = [rmap1.shape[i]*h1.workcell[i]/shp for i, shp in enumerate(h1.workarr.shape)]
    for _ in range(3): newcell.append(90.0)
    m1 = iotools.Map('fullmap.mrc')
    m1.arr = fullmap
    m1.cell = newcell
    m1.origin = [0, 0, 0]
    m1.write()
    #m1 = iotools.Map('fullmap.mrc')
    #m1.read()

    mm = iotools.Map('mask.mrc')
    mm.arr = rmask
    mm.cell = newcell
    mm.origin = [0, 0, 0]
    mm.write()
    #mm = iotools.Map('mask.mrc')
    #mm.read()

    tlist = []
    #for axis, fold in zip(axes, folds):
    #    # get the t_centroid
    #    rotcentre, t = grc.main('fullmap.mrc', axis, fold, 'mask.mrc', resol)
    #    #emmap1, rotcentre = get_rotation_center(m1, axis, fold, resol, mm)
    #    #print('rotation centre: ', rotcentre)
    #    #temp_t = np.subtract(rotcentre, emmap1.com1)
    #    #t_to_centroid = [temp_t[i]/emmap1.map_dim[i] for i in range(3)]
    #    #t = t_to_centroid
    #    tlist.append(t)

    tlist.append([0., 0., 0.])

    map1 = fullmap * rmask
    half1masked = rmap1 * rmask
    half2masked = rmap2 * rmask

    nx, ny, nz = map1.shape
    com = center_of_mass_density(map1)
    print("com:", com)
    box_centr = (nx // 2, ny // 2, nz // 2)
    half1_com_adjusted = half1masked #shift_density(half1masked, np.subtract(box_centr, com))
    half2_com_adjusted = half2masked #shift_density(half2masked, np.subtract(box_centr, com))
    #mask_com_adjusted = shift_density(rmask, np.subtract(box_centr, com))

    f1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(half1_com_adjusted)))
    f2 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(half2_com_adjusted)))

    nbin, res_arr, bin_idx, sgrid = restools.get_resolution_array(newcell, f1)

    fsc1, fsc2 = main(f_list=[f1, f2], 
         axes=axes,
         folds=folds,
         tlist=tlist,
         bin_idx=bin_idx,
         nbin=nbin)

    print("***** FSC Table *****")
    print("Bin#   Resol.   Original_FSC    SymAvg_FSC")
    for i in range(nbin):
        print( "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(i, res_arr[i], fsc1[i], fsc2[i]))


    plotter.plot_nlines(
        res_arr=res_arr, 
        list_arr=[fsc1, fsc2],
        curve_label=['original', 'averaged']
    )