"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import emda2.core as core
import numpy as np
import fcodes2 as fc
from emda2.config import *


def lowpassmap_ideal(fc, bin_idx, cbin):
    # ideal filter
    if not np.iscomplexobj(fc):
        fc = np.fft.fftshift(np.fft.fftn(fc))
    fout = core.restools.cut_resolution(fc, bin_idx, cbin)
    lowpmap = np.real(np.fft.ifftn(np.fft.ifftshift(fout)))
    return fout, lowpmap


def lowpassmap_butterworth(fc, sgrid, smax, order=4):
    if not np.iscomplexobj(fc):
        fc = np.fft.fftshift(np.fft.fftn(fc))
    order = 4  # order of the butterworth filter
    B = 1.0
    D = sgrid
    d = 1.0 / smax # smax in Ansgtrom units
    bwfilter = 1.0 / (
        1 + B * ((D / d) ** (2 * order))
    )  # calculate the butterworth filter
    fmap = fc * bwfilter
    lwpmap = np.real(np.fft.ifftn(np.fft.ifftshift(fmap)))
    return fmap, lwpmap

def set_array(arr, thresh=0.0):
    set2zero = False
    i = -1
    for ival in arr:
        i = i + 1
        if ival <= thresh and not set2zero:
            set2zero = True
        if set2zero:
            arr[i] = 0.0
    return arr

def interpolate_cc(data):
    from scipy.interpolate import RegularGridInterpolator

    linx = np.linspace(0, data.shape[0], data.shape[0])
    liny = np.linspace(0, data.shape[1], data.shape[1])
    linz = np.linspace(0, data.shape[2], data.shape[2])
    return RegularGridInterpolator(
        (linx, liny, linz), data, method="nearest")

def sphere_mask(nx):
    # Creating a sphere mask
    box_size = nx
    box_radius = nx // 2 -1
    center = [nx//2, nx//2, nx//2]
    print("boxsize: ", box_size, "boxradius: ", box_radius, "center:", center)
    radius = box_radius
    X, Y, Z = np.ogrid[:box_size, :box_size, :box_size]
    dist_from_center = np.sqrt(
        (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    )
    mask = dist_from_center <= radius
    return mask

def rotate_f(rotmat, f, interp='linear'):
    if len(f.shape) == 3:
        f = np.expand_dims(f, axis=3)
    f_rotated = _interp(rotmat, f, interp)
    return f_rotated

def _interp(RM, data, interp='linear'):
    assert len(data.shape) == 4
    ih, ik, il, n = data.shape
    if interp == "cubic":
        interp3d = fc.tricubic(RM, data, debug_mode, n, ih, ik, il)
    if interp == "linear":
        interp3d = fc.trilinear(RM, data, debug_mode, n, ih, ik, il)
    return interp3d

def shift_density(arr, shift):
    """Returns a shifted copy of the input array.

    Shift the array using spline interpolation (order=3). Same as Scipy
    implementation.

    Arguments:
        Inputs:
            arr: density as 3D numpy array
            shift: sequence. The shifts along the axes.

        Outputs:
            shifted_arr: ndarray. Shifted array
    """
    from scipy import ndimage

    # return ndimage.interpolation.shift(arr, shift)
    return ndimage.shift(arr, shift, mode="wrap")

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


def cut_resolution_for_linefit(f_list, bin_idx, res_arr, smax):
    # Making data for map fitting
    f_arr = np.asarray(f_list, dtype="complex")
    nx, ny, nz = f_list[0].shape
    cbin = cx = smax
    dx = int((nx - 2 * cx) / 2)
    dy = int((ny - 2 * cx) / 2)
    dz = int((nz - 2 * cx) / 2)
    cBIdx = bin_idx[dx: dx + 2 * cx, dy: dy + 2 * cx, dz: dz + 2 * cx]
    fout = fc.cutmap_arr(
        f_arr, bin_idx, cbin, 0, len(res_arr), nx, ny, nz, len(f_list)
    )[:, dx: dx + 2 * cx, dy: dy + 2 * cx, dz: dz + 2 * cx]
    return fout, cBIdx, cbin

def determine_ibin(bin_fsc, cutoff=0.15):
    bin_fsc = filter_fsc(bin_fsc)
    ibin = get_ibin(bin_fsc, cutoff)        
    i = 0
    while ibin < 5:
        cutoff -= 0.01
        ibin = get_ibin(bin_fsc, max([cutoff, 0.1]))
        i += 1
        if i > 100:
            print("Fit starting configurations are too far.")
            raise SystemExit()
    if ibin == 0:
        print("Fit starting configurations are too far.")
        raise SystemExit()
    return ibin

def filter_fsc(bin_fsc, thresh=0.05):
    bin_fsc_new = np.zeros(bin_fsc.shape, 'float')
    for i, ifsc in enumerate(bin_fsc):
        if ifsc >= thresh:
            bin_fsc_new[i] = ifsc
        else:
            if i > 1:
                break
    return bin_fsc_new

def get_ibin(bin_fsc, cutoff):
    # search from rear end
    ibin = 0
    for i, ifsc in reversed(list(enumerate(bin_fsc))):
        if ifsc > cutoff:
            ibin = i
            if ibin % 2 != 0:
                ibin = ibin - 1
            break
    return ibin