import numpy as np
import fcodes2
import emda2.core.iotools as iotools

debug_mode = 0

def get_map_power(fo, bin_idx, nbin, tol=1e-4):
    nx, ny, nz = fo.shape
    power_spectrum = fcodes2.calc_power_spectrum(
        fo, bin_idx, nbin, debug_mode, nx, ny, nz
    )
    return power_spectrum

def apply_bfactor_to_map(fmap, bf_arr, uc):
    nx, ny, nz = fmap.shape
    nbf = len(bf_arr)
    all_mapout = fcodes2.apply_bfactor_to_map(
        fmap, bf_arr, uc, debug_mode, nx, ny, nz, nbf
    )
    return all_mapout

def map2mtz(arr, uc, mtzname="map2mtz.mtz", resol=None):
    f1 = np.fft.fftshift(np.fft.fftn(np.transpose(arr)))
    iotools.write_3d2mtz(unit_cell=uc, mapdata=f1, outfile=mtzname, resol=resol)

def mtz2map(mtzname, map_size):
    from emda2.core.mtz import mtz2map
    arr, unit_cell = mtz2map(mtzname=mtzname, map_size=map_size)
    return arr, unit_cell


def interp_rgi(data):
    from scipy.interpolate import RegularGridInterpolator
    nx, ny, nz = data.shape
    arr = np.zeros(shape=(nx+8, ny+8, nz+8), dtype=type(data))
    arr[4:4+nx, 4:4+ny, 4:4+nz] = data
    linx = np.linspace(-(nx+8)//2, (nx+6)//2, nx+8)
    liny = np.linspace(-(ny+8)//2, (ny+6)//2, ny+8)
    linz = np.linspace(-(nz+8)//2, (nz+6)//2, nz+8)
    return RegularGridInterpolator(
        (linx, liny, linz), arr, method="linear")

def get_points(rm, nx, ny, nz):
    xx, yy, zz = np.mgrid[-nx//2 : nx//2, 
                          -ny//2 : ny//2, 
                          -nz//2 : nz//2]
    A = np.zeros(shape=(np.size(xx.flatten()), 3), dtype='float')
    A[:,0] = xx.flatten()
    A[:,1] = yy.flatten()
    A[:,2] = zz.flatten()
    points = np.dot(A, rm)
    points[:,0] = np.where(float(-nx//2) <= points[:,0], points[:,0], float(-nx//2))
    points[:,0] = np.where(float(nx//2) > points[:,0], points[:,0], float(nx//2-1))
    points[:,1] = np.where(float(-ny//2) <= points[:,1], points[:,1], float(-ny//2))
    points[:,1] = np.where(float(ny//2) > points[:,1], points[:,1], float(ny//2-1))
    points[:,2] = np.where(float(-nz//2) <= points[:,2], points[:,2], float(-nz//2))
    points[:,2] = np.where(float(nz//2) > points[:,2], points[:,2], float(nz//2-1))
    return points

def center_of_mass_density(arr):
    """Returns the center of mass of 3D density array.

    This function accepts density as 3D numpy array and caclulates the
    center-of-mass.

    Arguments:
        Inputs:
            arr: density as 3D numpy array

        Outputs:
            com: tuple, center-of-mass (x, y, z)
    """
    from scipy import ndimage

    return ndimage.measurements.center_of_mass(arr * (arr >= 0.0))


def map_output(arrlist, axis, angle, translation, mask=None):
    """
    This function returns rotated and translated copies of input maps
    in arrlist.
    Inputs:
        arrlist: list of ndarrays (maps)
            The same transformation will be applied on all maps.
        axis: rotation axis about which the maps are rotated. axis is normalised
            before being used.
        angle: angle by which the maps are rotated. 
            Note that angle should be given in Degrees. 
        translation: translation vector for maps.
            Note that translation should be given in fractionals 
            (direct output of EMDA axis refinement).
        mask: if given, input maps are maltiplied by this mask before other
            operations are carried out.
    
    Outputs:
        original_centered_maps: list of inputmaps centered at the center of box
            order of maps - fullmap, [halfmap1, halfmap2]
        transformed_maps: list of transformed maps
            order of maps - fullmap, [halfmap1, halfmap2]
    """
    import math
    import fcodes2 as fc
    from numpy.fft import fftshift, fftn, ifftshift, ifftn
    from emda2.core import quaternions
    from emda2.ext.utils import center_of_mass_density, shift_density, rotate_f

    # output lists
    original_centered_maps = []
    transformed_maps = []
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    q = quaternions.get_quaternion([list(axis), angle])
    rotmat = quaternions.get_RM(q)
    arr1 = arr2 = None
    if mask is not None:
        mask = mask * mask > 1e-4
    else:
        mask = 1.
    if len(arrlist) == 2:
        arr1 = arr1 * mask
        arr2 = arr2 * mask
        arr = (arr1 + arr2)
    elif len(arrlist) == 1:
        arr = arr * mask
    com = center_of_mass_density(arr)
    print('com: ', com)
    nx, ny, nz = arr.shape
    box_centr = (nx // 2, ny // 2, nz // 2)
    arr = shift_density(arr, np.subtract(box_centr, com))
    original_centered_maps.append(arr)
    fo = fftshift(fftn(fftshift(arr)))
    st = fc.get_st(nx, ny, nz, translation)[0]
    frt = rotate_f(rotmat, fo * st, interp="linear")[:, :, :, 0]
    data2write = np.real(ifftshift(ifftn(ifftshift(frt))))
    transformed_maps.append(data2write)
    if arr1 is not None:
        arr1 = shift_density(arr1, np.subtract(box_centr, com))
        arr2 = shift_density(arr2, np.subtract(box_centr, com))
        original_centered_maps.append(arr1)
        original_centered_maps.append(arr2)
        fo1 = fftshift(fftn(fftshift(arr1)))
        fo2 = fftshift(fftn(fftshift(arr2)))
        st = fc.get_st(nx, ny, nz, translation)[0]
        frt1 = rotate_f(rotmat, fo1 * st, interp="linear")[:, :, :, 0]
        frt2 = rotate_f(rotmat, fo2 * st, interp="linear")[:, :, :, 0]
        data2write1 = np.real(ifftshift(ifftn(ifftshift(frt1))))
        data2write2 = np.real(ifftshift(ifftn(ifftshift(frt2))))
        transformed_maps.append(data2write1)
        transformed_maps.append(data2write2) 
    return original_centered_maps, transformed_maps