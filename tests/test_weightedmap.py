import numpy as np
import emda2.emda_methods2 as em
from emda2.core import iotools, plotter
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--half1name', type=str, required=True, help='half1 file mrc/map')
parser.add_argument('--half2name', type=str, required=True, help='half2 file mrc/map')
#parser.add_argument('--maskname', type=str, required=False, default=None, help='mask file mrc/map')
args = parser.parse_args()

h1 = iotools.Map(name=args.half1name)
h1.read()
h2 = iotools.Map(name=args.half2name)
h2.read()
#mm = iotools.Map(name=args.maskname)
#mm.read()

nbin, res_arr, bin_idx, _ = em.get_binidx(h1.cell, h1.workarr)
weightedmap = em.emda_weightedmap(h1, h2, bin_idx, nbin, res_arr)

m2 = iotools.Map(name='emda_weightedmap.mrc')
m2.arr = weightedmap
m2.cell = h1.cell
m2.origin = h1.origin
m2.write()