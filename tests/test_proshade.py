# Apply b factor to map

import numpy as np
import emda2.emda_methods2 as em
from emda2.core import iotools, plotter, quaternions, restools
import math
import argparse, os
#from emda2.ext.sym import proshade_tools
from emda2.ext.utils import rotate_f


parser = argparse.ArgumentParser(description='Run Proshade on the map.')
parser.add_argument('--mapname', type=str, required=True, help='map1 file mrc/map')
args = parser.parse_args()

# C       +2      +0.96981   -0.21398   +0.11695     +3.14159      +0.64579      +0.06460

def transform(fo, axis, angle):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    q = quaternions.get_quaternion(list(axis), angle)
    rotmat = quaternions.get_RM(q)
    return rotate_f(rotmat, fo, interp="linear")[:, :, :, 0]


m1 = iotools.Map(args.mapname)
m1.read()

f1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(m1.workarr)))

nbin, res_arr, bin_idx, sgrid = restools.get_resolution_array(m1.workcell, f1)

# axis = np.array([0.96981, -0.21398, 0.11695], "float")
axis = np.array([0.0, 0.0, 1.0], "float")
rotated_f1 = transform(fo=f1, axis=axis, angle=180.)

rotated_map = np.real(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(rotated_f1))))

m2 = iotools.Map("rotated_c2_z.mrc")
m2.arr = rotated_map
m2.cell = m1.workcell
m2.write()