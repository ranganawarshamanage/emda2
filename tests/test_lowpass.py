# Lowpass filter map

import emda2.emda_methods2 as em
from emda2.core import iotools
import argparse, os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mapname', type=str, required=True, help='map file mrc/map')
parser.add_argument('--lowpass_resolution', type=float, required=True, help='Specify lowpass resolution')
parser.add_argument('--outputmapname', type=str, default=None, required=False, help='outputmap name mrc/map')
#parser.add_argument('--binfactor', type=float, required=True, help='Specify the bin factor')
args = parser.parse_args()


m1 = iotools.Map(name=args.mapname)
m1.read()

_, lwp = em.lowpass_map(
    uc=m1.workcell, arr1=m1.workarr, resol=args.lowpass_resolution, filter="butterworth")

if args.outputmapname is None:
    basename = os.path.basename(args.mapname)
    outputmapname = basename[:-4] + "_emda_lowpass_%dA.mrc" % args.lowpass_resolution

m2 = iotools.Map(name=args.outputmapname)
croppedImage = iotools.cropimage(arr=lwp, tdim=m1.arr.shape)
m2.cell = m1.cell
m2.arr = croppedImage
m2.axorder = m1.axorder
m2.write()