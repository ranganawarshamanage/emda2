# Compute Map power spectrum

import emda2.emda_methods2 as em
from emda2.core import iotools, fft, plotter
import argparse

parser = argparse.ArgumentParser(description='Compute Map power spectrum')
parser.add_argument('--mapname', type=str, required=True, help='map file mrc/map')
args = parser.parse_args()


m1 = iotools.Map(name=args.mapname)
m1.read()


nbin, res_arr, bin_idx, sgrid = em.get_binidx(m1.workcell, m1.workarr)
power_spectrum = em.get_map_power(
    fo=fft.to_f(m1.workarr),
    bin_idx=bin_idx, 
    nbin=nbin
)

plotter.plot_nlines_log(
    res_arr=res_arr, 
    list_arr=[power_spectrum], 
    labels=["power"],
    ylabel="Log(Power)",
    )

print("Resolution   bin     Power")
for i in range(len(res_arr)):
    print("{:.2f} {:.4f}".format(res_arr[i], power_spectrum[i]))  