import emda2.emda_methods2 as em
from emda2.core import iotools
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mapname', type=str, required=True, help='map file mrc/map')
parser.add_argument('--maskname', type=str, required=True, help='mask file mrc/map')
#parser.add_argument('--binfactor', type=float, required=True, help='Specify the bin factor')
args = parser.parse_args()


m1 = iotools.Map(name=args.mapname)
m1.read()
mm = iotools.Map(name=args.maskname)
mm.read()

arr = em.applymask(m1, mm)

m2 = iotools.Map(name='masked_map.mrc')
m2.arr = arr
m2.cell = m1.cell
m2.origin = m1.origin
m2.write()