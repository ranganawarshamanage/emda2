import emda2.emda_methods2 as em
from emda2.core import iotools
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mapname', type=str, required=True, help='map file mrc/map')
parser.add_argument('--binary_threshold', type=float, required=True, help='binary threshold to use')
parser.add_argument('--maskname', type=str, required=False, default="emda_mapmask.mrc", help='Specify the bin factor')
args = parser.parse_args()


m1 = iotools.Map(name=args.mapname)
m1.read()
print(m1.cell)
print(m1.arr.shape)

masklist = em.mask_from_map_connectedpixels(m1, binthresh=args.binary_threshold)

m2 = iotools.Map(name=args.maskname)
m2.arr = masklist[0]
m2.cell = m1.cell
m2.origin = m1.origin
m2.write()

