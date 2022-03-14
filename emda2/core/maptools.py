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