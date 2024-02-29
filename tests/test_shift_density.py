from emda2.core import iotools
from emda2.ext.utils import shift_density
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mapname', type=str, required=True, help='map file mrc/map')
parser.add_argument(
    "--translation",
    required=True,
    default=[0.0, 0.0, 0.0],
    nargs="+",
    type=float,
    help="translation vec. in Angstrom. eg 1.0 0.0 0.0",
)
parser.add_argument('--outmapname', type=str, required=False, default="emda_shifted.mrc", help='Specify the output mapname')
args = parser.parse_args()


m1 = iotools.Map(name=args.mapname)
m1.read()
print(m1.cell)
print(m1.arr.shape)
pixsize = m1.cell[0] / m1.arr.shape[0]

shift_px = [t/pixsize for t in args.translation]

m2 = iotools.Map(name=args.outmapname)
m2.arr = shift_density(m1.arr, shift=shift_px)
m2.cell = m1.cell
m2.origin = m1.origin
m2.write()