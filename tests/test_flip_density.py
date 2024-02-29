import numpy as np
from emda2.core import iotools
from emda2.ext.utils import shift_density
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mapname', type=str, required=True, help='map file mrc/map')
parser.add_argument(
    "--flip_by",
    required=True,
    default="Z",
    type=str,
    help="Axis for flipping e.g. Z (default)",
)
parser.add_argument('--outmapname', type=str, required=False, default="emda_flipped.mrc", help='Specify the output mapname')
args = parser.parse_args()


m1 = iotools.Map(name=args.mapname)
m1.read()
print(m1.cell)
print(m1.arr.shape)
pixsize = m1.cell[0] / m1.arr.shape[0]

if args.flip_by == 'X':
    flip_ax = 0
elif args.flip_by == 'Y':
    flip_ax = 1
elif args.flip_by == 'Z':
    flip_ax = 2
else:
    raise SystemExit("Only permmited options - X, Y, Z")

m2 = iotools.Map(name=args.outmapname)
m2.arr = np.flip(m1.arr, axis=flip_ax)
m2.cell = m1.cell
m2.origin = m1.origin
m2.write()