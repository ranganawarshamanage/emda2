import numpy as np
import emda2.emda_methods2 as em
from emda2.core import iotools, plotter
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--map1name', type=str, required=True, help='map1 file mrc/map')
parser.add_argument('--map2name', type=str, required=True, help='map2 file mrc/map')
parser.add_argument('--maskname', type=str, required=False, default=None, help='mask file mrc/map')
args = parser.parse_args()

m1 = iotools.Map(name=args.map1name)
m1.read()
m2 = iotools.Map(name=args.map2name)
m2.read()

if args.maskname is not None:
    mm = iotools.Map(name=args.maskname)
    mm.read()
    mask = mm.workarr
else:
    mask = 1.0

nbin, res_arr, bin_idx, _ = em.get_binidx(m1.cell, m1.workarr)

twomapfsc = em.fsc(f1=np.fft.fftshift(np.fft.fftn(m1.workarr * mask)),
                f2=np.fft.fftshift(np.fft.fftn(m2.workarr * mask)),
                bin_idx=bin_idx,
                nbin=nbin)

print("---- EMDA FSC ----")
for i, fsc in enumerate(twomapfsc):
    print(i, res_arr[i], fsc)

plotter.plot_nlines(res_arr=res_arr,
                    list_arr=[twomapfsc],
                    mapname="emda_fsc",)