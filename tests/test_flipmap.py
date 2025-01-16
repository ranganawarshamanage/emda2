# Flip map by define plane

import emda2.emda_methods2 as em
from emda2.core import iotools
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mapname', type=str, required=True, help='map file mrc/map')
parser.add_argument('--axis', type=str, required=True, help='Specify axis to flip the map')
args = parser.parse_args()


m1 = iotools.Map(name=args.mapname)
m1.read()

flipped_arr = em.flip_arr(m1.workarr, args.axis)

m2 = iotools.Map(f"flipped_map_axis{args.axis}.mrc")
m2.arr = flipped_arr
m2.cell = m1.cell
m2.write()


