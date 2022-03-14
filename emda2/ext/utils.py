"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import emda2.core as core
import numpy as np
import fcodes2
from emda.config import *


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