import emda2.emda_methods2 as em
from emda2.core import iotools
import argparse, os
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mapname', type=str, required=True, help='map file mrc/map')
parser.add_argument('--dimension', type=int, required=True, help='Specify dimension for the cube')
parser.add_argument('--outputmapname', type=str, default='emda_cubic.mrc', required=False, help='outputmap name mrc/map')
args = parser.parse_args()


m1 = iotools.Map(name=args.mapname)
m1.read()
nx, ny, nz = m1.workarr.shape

cdim = args.dimension

try:

    cubic_arr = np.zeros((cdim, cdim, cdim), dtype=float)

    # coordinates of the center
    dx = (cdim - m1.workarr.shape[0]) // 2
    dy = (cdim - m1.workarr.shape[1]) // 2
    dz = (cdim - m1.workarr.shape[2]) // 2

    # place the map at the center of the cubic array
    cubic_arr[
        dx:dx+m1.workarr.shape[0], 
        dy:dy+m1.workarr.shape[1], 
        dz:dz+m1.workarr.shape[2]] = m1.workarr

    # new metadata
    pixsize = m1.workcell[0] / nx
    cell = [cdim * pixsize for _ in range(3)] + [90., 90., 90.]

    # output map
    m2 = iotools.Map(name=args.outputmapname)
    m2.cell = cell
    m2.arr = cubic_arr
    m2.axorder = m1.axorder
    m2.write()

except ValueError as e:
    print(e)