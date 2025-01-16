import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift


def to_map(fo):
    return np.real(ifftshift(ifftn(ifftshift(fo))))

def to_f(rho):
    return fftshift(fftn(fftshift(rho)))