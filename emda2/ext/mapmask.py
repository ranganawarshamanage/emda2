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
    lwp = (lwp - np.mean(lwp)) / np.std(lwp)
    if binary_threshold is None:
        binary_threshold = np.amax(lwp) / 10
    arr = lwp > binary_threshold
    blobs = arr
    radius = max(blobs.shape) // 2
    gmask = globular_mask(arr=blobs, radius=radius, com=True)
    blobs = blobs * gmask
    blobs_labels, nlabels = measure.label(blobs, background=0, connectivity=blobs.ndim, return_num=True)
    #print('blob assignment done')
    regionprops = measure.regionprops(blobs_labels)
    bvol = 0
    for i in range(nlabels):
        if bvol < regionprops[i].area:
            bvol = regionprops[i].area
            largest_blob = i
        else:
            continue
    mask = blobs * (blobs_labels == largest_blob+1)
    nmask = binary_closing(mask * gmask)
    nmask = binary_dilation(nmask)
    return nmask, arr

def main(imap):
    maskname = imap[:-4] + "_mapmask.mrc"
    #m = re.search('emd_(.+?).map', imap)
    #maskname = m.group(1) + "_mapmask.mrc"
    #pltname = m.group(1) + "_rad.eps"
    m1 = iotools.Map(name=imap)
    m1.read()
    mask, lwp = mapmask_connectedpixels(m1=m1)
    mout = iotools.Map(name=maskname)
    mout.arr = mask
    mout.cell = m1.workcell
    mout.origin = m1.origin
    mout.write()
    """ mout = iotools.Map(name='lwp.mrc')
    mout.arr = lwp
    mout.cell = m1.workcell
    mout.origin = m1.origin
    mout.write() """


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
 
