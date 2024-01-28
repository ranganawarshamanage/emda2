"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import math
import emda2.core as core
from emda2.core import iotools, maptools
import emda2.emda_methods2 as em
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
    D = sgrid
    d = 1.0 / smax  # smax in Ansgtrom units
    # butterworth filter
    bwfilter = 1.0 / np.sqrt(1 + ((D / d) ** (2 * order)))
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
    return RegularGridInterpolator((linx, liny, linz), data, method="nearest")


def sphere_mask(nx):
    # Creating a sphere mask
    box_size = nx
    box_radius = nx // 2 - 1
    center = [nx // 2, nx // 2, nx // 2]
    print("boxsize: ", box_size, "boxradius: ", box_radius, "center:", center)
    radius = box_radius
    X, Y, Z = np.ogrid[:box_size, :box_size, :box_size]
    dist_from_center = np.sqrt(
        (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    )
    mask = dist_from_center <= radius
    return mask


def rotate_f(rotmat, f, interp="linear"):
    if len(f.shape) == 3:
        f = np.expand_dims(f, axis=3)
    f_rotated = _interp(rotmat, f, interp)
    return f_rotated


def _interp(RM, data, interp="linear"):
    assert len(data.shape) == 4
    ih, ik, il, n = data.shape
    if interp == "cubic":
        interp3d = fc.tricubic(RM, data, debug_mode, n, ih, ik, il)
    if interp == "linear":
        interp3d = fc.trilinear(RM, data, debug_mode, n, ih, ik, il)
    return interp3d


def rotate_f_within_sphere(rotmat, f, bin_idx, ibin):
    """Returns a rotated copy of f

    f is a stack of 3D complex arrays. This method rotates
    each complex array by the given rotation matrix.
    ibin defines the radius of the sphere to be roated. All the
    Fourier coefficinets are roated within that sphere.
    """
    if len(f.shape) == 3:
        f = np.expand_dims(f, axis=3)
    nx, ny, nz, ncopies = f.shape
    return fc.trilinear_sphere(
        rotmat, f, bin_idx, 0, ibin, ncopies, nx, ny, nz
    )


def get_xyz_sum(xyz):
    xyz_sum = np.zeros(shape=(6), dtype="float")
    n = -1
    for i in range(3):
        for j in range(3):
            if i == 0:
                sumxyz = np.sum(xyz[i] * xyz[j])
            elif i > 0 and j >= i:
                sumxyz = np.sum(xyz[i] * xyz[j])
            else:
                continue
            n = n + 1
            xyz_sum[n] = sumxyz
    return xyz_sum


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
    cBIdx = bin_idx[dx : dx + 2 * cx, dy : dy + 2 * cx, dz : dz + 2 * cx]
    fout = fc.cutmap_arr(
        f_arr, bin_idx, cbin, 0, len(res_arr), nx, ny, nz, len(f_list)
    )[:, dx : dx + 2 * cx, dy : dy + 2 * cx, dz : dz + 2 * cx]
    return fout, cBIdx, cbin


def filter_fsc(bin_fsc, thresh=0.05):
    bin_fsc_new = np.zeros(bin_fsc.shape, "float")
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


def determine_ibin(bin_fsc, cutoff=0.15):
    return get_ibin(filter_fsc(bin_fsc), cutoff)


def get_avg_fsc(binfsc, bincounts):
    fsc_avg = 0.
    fsc_filtered = filter_fsc(bin_fsc=binfsc, thresh=0.0)
    if np.sum(fsc_filtered) > 0.:
        fsc_avg = np.average(
            a=fsc_filtered[np.nonzero(fsc_filtered)],
            weights=bincounts[np.nonzero(fsc_filtered)],
        )
    return fsc_avg


def rebox2cube(arr):
    nx, ny, nz = arr.shape
    maxdim = np.max(arr.shape)
    if maxdim % 2 != 0:
        maxdim += 1
    dx = (maxdim - nx) // 2
    dy = (maxdim - ny) // 2
    dz = (maxdim - nz) // 2
    newarr = np.zeros((maxdim, maxdim, maxdim), "float")
    newarr[dx : dx + nx, dy : dy + ny, dz : dz + nz] = arr
    return newarr


def rebox_using_radius(arr, padwidth=10, rad=None):
    """
    Returns a cubic array.

    Inputs:
        arr: ndarray of the map
        rad: int, half legnth (in pixels) of the box to be created.
            if not given, helf of the length of the given box is taken
        padwidth: int, number of pixels to be padded. default to 10

    Outputs:
        newarr: reboxed array
    """
    arrshape = np.array(arr.shape, "int")
    if np.sum(arrshape - np.amax(arrshape)) != 0:
        arr = rebox2cube(arr)
    nx, ny, nz = arr.shape
    if rad is None:
        rad = nx // 2
    if rad <= 0:
        raise SystemExit("rad CANNOT BE negative or zero!")
    assert rad <= nx // 2
    x1 = y1 = z1 = nx // 2 - rad
    x2 = y2 = z2 = x1 + 2 * rad
    dimz = z2 - z1
    dimy = y2 - y1
    dimx = x2 - x1
    dim = np.max([dimz, dimy, dimx])
    if dim % 2 != 0:
        dim += 1
    newarr = np.zeros(
        (dim + padwidth * 2, dim + padwidth * 2, dim + padwidth * 2), "float"
    )
    dx = (dim - dimx) // 2
    dy = (dim - dimy) // 2
    dz = (dim - dimz) // 2
    dx += padwidth
    dy += padwidth
    dz += padwidth
    newarr[dz : dz + dimz, dy : dy + dimy, dx : dx + dimx] = arr[
        z1:z2, y1:y2, x1:x2
    ]
    return newarr


""" def rebox_using_mask(arr, mask, padwidth=10):
    mask = mask * (mask > 1.e-5)
    i, j, k = np.nonzero(mask)
    z2, y2, x2 = np.max(i), np.max(j), np.max(k)
    z1, y1, x1 = np.min(i), np.min(j), np.min(k)
    dimz = z2 - z1
    dimy = y2 - y1
    dimx = x2 - x1
    dim = np.max([dimz, dimy, dimx])
    if dim % 2 != 0:
        dim += 1
    newarr  = np.zeros((dim+padwidth*2, dim+padwidth*2, dim+padwidth*2), 'float')
    newmask = np.zeros((dim+padwidth*2, dim+padwidth*2, dim+padwidth*2), 'float')
    print((dim - dimz), (dim - dimy), (dim - dimx))
    dz = (dim - dimz) // 2
    dy = (dim - dimy) // 2
    dx = (dim - dimx) // 2
    dz += padwidth
    dy += padwidth
    dx += padwidth
    newarr[dz:dz+dimz, dy:dy+dimy, dx:dx+dimx] = arr[z1:z2, y1:y2, x1:x2]
    newmask[dz:dz+dimz, dy:dy+dimy, dx:dx+dimx] = mask[z1:z2, y1:y2, x1:x2]
    return newarr, newmask """


def rebox_using_mask(arr, mask, mask_origin, padwidth=10):
    mask = mask * (mask > 1.0e-5)
    mo1, mo2, mo3 = mask_origin
    i, j, k = np.nonzero(mask)
    z2, y2, x2 = np.max(i), np.max(j), np.max(k)
    # z2, y2, x2 = z2+mo1, y2+mo2, x2+mo3
    z1, y1, x1 = np.min(i), np.min(j), np.min(k)
    # z1, y1, x1 = z1+mo1, y1+mo2, x1+mo3
    dimz = z2 - z1
    dimy = y2 - y1
    dimx = x2 - x1
    dim = np.max([dimz, dimy, dimx])
    if dim % 2 != 0:
        dim += 1
    newarr = np.zeros(
        (dim + padwidth * 2, dim + padwidth * 2, dim + padwidth * 2), "float"
    )
    newmask = np.zeros(
        (dim + padwidth * 2, dim + padwidth * 2, dim + padwidth * 2), "float"
    )
    print((dim - dimz), (dim - dimy), (dim - dimx))
    dz = (dim - dimz) // 2
    dy = (dim - dimy) // 2
    dx = (dim - dimx) // 2
    dz += padwidth
    dy += padwidth
    dx += padwidth
    #
    z1arr = z1  # +mo1
    z2arr = z2  # +mo1
    y1arr = y1  # +mo2
    y2arr = y2  # +mo2
    x1arr = x1  # +mo3
    x2arr = x2  # +mo3
    nx, ny, nz = arr.shape
    try:
        assert z1arr < z2arr <= nz
        assert y1arr < y2arr <= ny
        assert x1arr < x2arr <= nx
        # non-cubic box
        # newarr = arr[z1+mo1:z2+mo1, y1+mo2:y2+mo2, x1+mo3:x2+mo3]
        # newmask = mask[z1:z2, y1:y2, x1:x2]
        # cubic box
        newarr[dz : dz + dimz, dy : dy + dimy, dx : dx + dimx] = arr[
            z1arr:z2arr, y1arr:y2arr, x1arr:x2arr
        ]
        newmask[dz : dz + dimz, dy : dy + dimy, dx : dx + dimx] = mask[
            z1:z2, y1:y2, x1:x2
        ]
        return newarr, newmask
    except Exception as e:
        print(e)


def rebox_using_model(imap, model):
    """
    Reboxes map using a coordinate generated mask.
    This includes following steps:
    1. generate mask from model
    2. rebox map using that mask
    3. rebox model using that mask

    Inputs:
        imap: mapname
        model: modelname

    Outputs:
        reboxed_arr: ndarray, reboxed map
        reboxed_mask: ndarray, reboxed mask
        reboxed_model: revoxed model 'emda_reboxed_model.cif' is output
    """
    mm = em.mask_from_atomic_model(mapname=imap, modelname=model)
    m1 = iotools.Map(name=imap)
    m1.read()
    reboxed_arr, reboxed_mask = rebox_using_mask(arr=m1.workarr, mask=mm.arr)
    # model rebox
    maptools.model_rebox(mask=mm.arr, mmcif_file=model, uc=m1.cell)
    return reboxed_arr, reboxed_mask


def vec2string(vec):
    return " ".join(("% .3f" % x for x in vec))


def is_prime(n):
    """
    https://stackoverflow.com/questions/15285534/
    isprime-function-for-python-language
    """
    if n == 2 or n == 3:
        return True
    if n < 2 or n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    r = int(n**0.5)
    f = 5
    while f <= r:
        if n % f == 0:
            return False
        if n % (f + 2) == 0:
            return False
        f += 6
    return True


def normalise_axis(axis):
    ax = np.asarray(axis, "float")
    return ax / math.sqrt(np.dot(ax, ax))