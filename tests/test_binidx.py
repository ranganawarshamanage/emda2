
# Generate the 3D grid of indices for down stream map calculations

import emda2.emda_methods2 as em
from emda2.core import iotools
import argparse

parser = argparse.ArgumentParser(description='Generate 3D index grid')
parser.add_argument('--mapname', type=str, required=True, help='map file mrc/map')
args = parser.parse_args()

# Read the map and generate the map-object
m1 = iotools.Map(args.mapname)
m1.read()

# Generate the 3D grid of indices

# res_arr is a 1D numpy array of resolution resolution where
# res_arr[0] - lowest resolution
# res_arr[-1] - highest resolution

# bin_idx is a Numpy 3D array (int). Its shells radiating from its centre
# has resolution index. centre shell has value 0 and the outermost
# shell has the highest value = len(res_arr) - 1
# Its corners has the value -100

# sgrid is also a 3D numpy array (float) which is the result of mapping
# res_arr into the bin_idx

nbin, res_arr, bin_idx, sgrid = em.get_binidx(
    cell=m1.workcell,
    arr=m1.workarr
)
