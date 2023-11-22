"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

import emda2.emda_methods2 as em
import numpy as np
import sys, re
from emda2.core import iotools, maptools


def globular_mask(arr, radius=None, com=False):  
    import math  
    box_size = max(arr.shape)
    nx, ny, nz = arr.shape
    box_radius = box_size // 2 - 1
    if radius is not None:
        if radius < box_radius:
            box_radius = radius
    if com:
        com1 = maptools.center_of_mass_density(arr)
        if math.isnan(com1[0]):
            center = [nx//2, ny//2, nz//2]
        else:
            center = [int(com1[0]), int(com1[1]), int(com1[2])]
    else:
        center = [nx//2, ny//2, nz//2]
    # Creating a sphere mask
    #print("boxsize: ", box_size, "boxradius: ", box_radius, "center:", center)
    #radius = box_radius
    X, Y, Z = np.ogrid[:nx, :ny, :nz]
    dist_from_center = np.sqrt(
        (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    )
    mask = dist_from_center <= box_radius
    return mask

def binary_closing(binary_arr, size=9):
    from scipy.ndimage.morphology import binary_closing
    selem = np.ones((size, size, size), 'int')
    newarr = binary_closing(binary_arr, structure=selem, iterations=1)
    return newarr

def binary_dilation(binary_arr, size=5):
    from scipy.ndimage.morphology import binary_dilation
    selem = np.ones((size, size, size), 'int')
    newarr = binary_dilation(binary_arr, structure=selem, iterations=1)
    return newarr

def mapmask_connectedpixels(m1, binary_threshold=None):
    from skimage import measure

    _, lwp = em.lowpass_map(uc=m1.workcell, arr1=m1.workarr, resol=15, filter="butterworth")
    if binary_threshold is None:
        lwp = (lwp - np.mean(lwp)) / np.std(lwp)
        binary_threshold = np.amax(lwp) / 20
    arr = lwp > binary_threshold
    blobs = arr
    radius = max(blobs.shape) // 2
    gmask = globular_mask(arr=blobs, radius=radius, com=True)
    blobs = blobs * gmask
    blobs_labels, nlabels = measure.label(blobs, background=0, connectivity=blobs.ndim, return_num=True)
    #print('blob assignment done')
    regionprops = measure.regionprops(blobs_labels)
    blob_number = []
    blob_area = []
    bsum = 0
    for i in range(nlabels):
        blob_number.append(i+1)
        blob_area.append(regionprops[i].area)
        bsum += regionprops[i].area

    from more_itertools import sort_together
    sblob_area, sblob_number = sort_together([blob_area, blob_number], reverse=True)
    bnum_highvol = []
    bsum_highvol = []
    rmsd_list = []
    for i in range(nlabels):
        vol_frac = sblob_area[i]/bsum
        if vol_frac >= 0.05:
            bnum_highvol.append(sblob_number[i])
            xx = m1.workarr * (blobs_labels == sblob_number[i])
            bsum_highvol.append(np.sum(xx))
            rmsd = np.sqrt(np.mean((xx - np.mean(xx))**2))
            rmsd_list.append(rmsd)
            print(sblob_number[i], np.sum(xx), vol_frac, rmsd)
    srmsd_list, sbnum_highvol = sort_together([rmsd_list, bnum_highvol], reverse=True)
    mask_list = []
    from emda2.ext.maskmap_class import make_soft
    for num in sbnum_highvol:
        mask = blobs * (blobs_labels == num)
        nmask = binary_closing(mask * gmask)
        nmask = binary_dilation(nmask) 
        nmask = make_soft(nmask, 3)
        mask_list.append(nmask)    
    return mask_list, lwp

def main(imap, imask=None):
    if imask is None:
        imask = imap[:-4] + "_mapmask.mrc"
    m1 = iotools.Map(name=imap)
    m1.read()
    masklist, lwp = mapmask_connectedpixels(m1=m1)
    mout = iotools.Map(name=imask)
    mout.arr = masklist[0]
    mout.cell = m1.workcell
    mout.origin = m1.origin
    mout.write()    
    """ mout = iotools.Map(name='lwp.mrc')
    mout.arr = lwp
    mout.cell = m1.workcell
    mout.origin = m1.origin
    mout.write() """



######### BEGIN - Mask from Halfmaps #########

def find_background(h1, h2):
    from emda2.core import restools
    from scipy.signal import fftconvolve

    _, lwp1 = em.lowpass_map(uc=h1.workcell, arr1=h1.workarr, resol=15, filter="butterworth")
    _, lwp2 = em.lowpass_map(uc=h2.workcell, arr1=h2.workarr, resol=15, filter="butterworth")
    noise = lwp1 - lwp2
    #noise = h1.workarr - h2.workarr
    kern = restools.create_soft_edged_kernel_pxl(5)
    loc3_A = fftconvolve(noise, kern, "same")
    loc3_A2 = fftconvolve(noise * noise, kern, "same")
    var3_A = loc3_A2 - loc3_A ** 2
    noise_std = np.sqrt(var3_A * (var3_A > np.amax(var3_A)/100))
    return lwp1, np.amax(noise_std)

def create_mapmask_islandlabelling2(m1, thresh):
    from skimage import measure
    _, lwp = em.lowpass_map(uc=m1.workcell, arr1=m1.workarr, resol=15, filter="butterworth")
    print('x y: ', np.amax(lwp))
    arr = lwp >= thresh
    blobs = arr
    radius = max(blobs.shape) // 2
    gmask = globular_mask(arr=blobs, radius=radius, com=True)
    blobs = blobs * gmask
    blobs_labels, nlabels = measure.label(blobs, background=0, connectivity=blobs.ndim, return_num=True)
    print('blob assignment done')
    regionprops = measure.regionprops(blobs_labels)
    # new code 13June2022
    blob_number = []
    blob_area = []
    bsum = 0
    for i in range(nlabels):
        blob_number.append(i+1)
        blob_area.append(regionprops[i].area)
        bsum += regionprops[i].area

    from more_itertools import sort_together
    sblob_area, sblob_number = sort_together([blob_area, blob_number], reverse=True)
    bnum_highvol = []
    bsum_highvol = []
    rmsd_list = []
    for i in range(nlabels):
        vol_frac = sblob_area[i]/bsum
        if vol_frac >= 0.05:
            bnum_highvol.append(sblob_number[i])
            xx = m1.workarr * (blobs_labels == sblob_number[i])
            bsum_highvol.append(np.sum(xx))
            rmsd = np.sqrt(np.mean((xx - np.mean(xx))**2))
            rmsd_list.append(rmsd)
            print(sblob_number[i], np.sum(xx), vol_frac, rmsd)
    #sbsum_highvol, sbnum_highvol = sort_together([bsum_highvol, bnum_highvol], reverse=True)
    print('sorting...')
    srmsd_list, sbnum_highvol = sort_together([rmsd_list, bnum_highvol], reverse=True)
    mask_list = []
    from emda2.ext.maskmap_class import make_soft
    for num in sbnum_highvol:
        print('making masks...')
        mask = blobs * (blobs_labels == num)
        nmask = binary_closing(mask * gmask)
        nmask = binary_dilation(nmask, 9) 
        nmask = make_soft(nmask, 3)
        mask_list.append(nmask)    
    return mask_list

def mask_from_halfmaps(half1, maskname=None):
    half2 = half1.replace("half_map_1", "half_map_2")
    h1 = iotools.Map(half1)
    h1.read()
    h2 = iotools.Map(half2)
    h2.read()
    lwp1, thresh = find_background(h1, h2)
    h1.workarr = lwp1
    print('threshold: ', thresh)
    masklist = create_mapmask_islandlabelling2(m1=h1, thresh=thresh)
    """ for i, mask in enumerate(masklist):
        maskname1 = "manual_emda_mask_%s.mrc" %str(i)
        mout = iotools.Map(name=maskname1)
        mout.arr = mask
        mout.cell = h1.workcell
        mout.origin = h1.origin
        mout.write() """
    if maskname is None:
        maskname = "manual_emda_mask_%s.mrc" %str(0)
    maskname1 = maskname
    mout = iotools.Map(name=maskname1)
    mout.arr = masklist[0]
    mout.cell = h1.workcell
    mout.origin = h1.origin
    mout.write()    

######### END - Mask from Halfmaps #########




if __name__=="__main__":
    #imap = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-0439/emd_0439.map"
    #imap = "/Users/ranganaw/MRC/REFMAC/EMD-6952/map/emd_6952.map"
    #imap = "/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3650/emd_3650_half_map_1.map"
    #imap = "/Users/ranganaw/MRC/REFMAC/EMD-0011/emd_0011.map"
    #imap = "/Users/ranganaw/MRC/REFMAC/beta_galactosidase/EMD-10563/emd_10563.map"
    #imap = "/Users/ranganaw/MRC/REFMAC/Jan_maps/postprocess_185.mrc"
    #imap = "/Users/ranganaw/MRC/REFMAC/PaaZ/EMD-9874/emd_9874.map"

    imap = sys.argv[1]
    main(imap)
 

