# Resample map
import emda2.emda_methods2 as em
from emda2.core import iotools
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mapname', type=str, required=True, help='map file mrc/map')
parser.add_argument('--target_pix', type=float, required=True, help='target pixel size')
parser.add_argument('--target_dim', type=int, nargs="+", default=None, required=False, help='target dimensions')
args = parser.parse_args()

mapobj1 = iotools.Map(name=args.mapname)
mapobj1.read()

if args.target_dim is None:
    target_dim = list(mapobj1.workarr.shape)
else:
    target_dim = args.target_dim

resampled_arr  = em.resample_data(curnt_pix=[mapobj1.workcell[i]/shape for i, shape in enumerate(mapobj1.workarr.shape)],
                                 targt_pix=[args.target_pix,args.target_pix,args.target_pix],
                                 arr=mapobj1.workarr,
                                 targt_dim=target_dim,
                                  )

tp = [args.target_pix, args.target_pix, args.target_pix]

mapobj = iotools.Map("resampled.mrc")
mapobj.arr = resampled_arr
mapobj.cell = [tp[i]*shape for i, shape in enumerate(resampled_arr.shape)]
mapobj.origin = mapobj1.origin
mapobj.axorder = mapobj1.axorder
mapobj.write()
