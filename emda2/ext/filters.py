"""
This is a difference filter to estimate noise from single map

"""
import numpy as np
import emda2.emda_methods2 as em
from scipy import signal

def difference_filter(arr):
    """
    This is a difference filter to estimate noise.
    Filter size 3 x 3 x 3.
    """
    npix = 3
    w = np.zeros((npix,npix,npix), 'float')
    occ = 0.85
    w = w - occ
    w[1][1][0] = -1
    w[0][1][1] = -1
    w[1][0][1] = -1
    w[1][1][1] = 6 + occ * 21
    w[1][2][1] = -1
    w[2][1][1] = -1
    w[1][1][2] = -1
    w = w / 27
    return signal.fftconvolve(arr, w, 'same')

def mean_filter(arr, size=3):
    """
    Mean filter to estimate mean field
    """
    npix = size
    w = np.ones((npix,npix,npix), 'float')
    w[npix//2][npix//2][npix//2] = 0.
    w = w / (npix*npix*npix - 1)
    return signal.fftconvolve(arr, w, 'same')

def median_filter(arr, size=3):
    """
    Median filter - Scipy implementation
    """
    npix = size
    from scipy.ndimage import median_filter
    return median_filter(input=arr, size=npix, mode='reflect')

def variances(arr, bin_idx, nbin):
    noise = filter(arr)
    psn = em.get_map_power(
        fo=np.fft.fftshift(np.fft.fftn(noise)),
        bin_idx=bin_idx, nbin=nbin)
    pst = em.get_map_power(
        fo=np.fft.fftshift(np.fft.fftn(arr)),
        bin_idx=bin_idx, nbin=nbin)
    pss = pst - psn
    return pss, psn