# Normalise Fourier coefficients within resolution bins

import emda2.emda_methods2 as em
from emda2.core import iotools, fft
import argparse

parser = argparse.ArgumentParser(description='Normalise Fourier coefficients')
parser.add_argument('--mapname', type=str, required=True, help='map file mrc/map')
args = parser.parse_args()

# Read the map and generate the map-object
m1 = iotools.Map(args.mapname)
m1.read()

nbin, res_arr, bin_idx, sgrid = em.get_binidx(
    cell=m1.workcell,
    arr=m1.workarr
)

normalised_f = em.get_normalised_f(
    fo=fft.to_f(m1.workarr), 
    bin_idx=bin_idx, 
    nbin=nbin
)

normalised_map = fft.to_map(normalised_f)