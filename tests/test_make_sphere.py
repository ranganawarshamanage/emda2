# create a sphere and output as a mrc file

import numpy as np
import emda2.emda_methods2 as em
from emda2.core import iotools
import argparse
from emda2.ext.maskmap_class import make_soft


parser = argparse.ArgumentParser(description='')
parser.add_argument('--mapname', type=str, required=True, help='map file mrc/map')
parser.add_argument('--radius', type=float, required=True, help='radius in Angstroms')
parser.add_argument('--offset_x', type=float, required=False, default=0.0, help='offset along x in Angstroms')
parser.add_argument('--offset_y', type=float, required=False, default=0.0, help='offset along y in Angstroms')
parser.add_argument('--offset_z', type=float, required=False, default=0.0, help='offset along z in Angstroms')

args = parser.parse_args()


def create_cylinder_array(radius, height, length, resolution=64):
    # Set up a grid
    x = np.linspace(-length/2, length/2, resolution)
    y = np.linspace(-radius, radius, resolution)
    z = np.linspace(0, height, resolution)
    x, y, z = np.meshgrid(x, y, z)

    # Create a 3D array representing the cylinder
    cylinder_array = np.zeros_like(x, dtype=int)
    cylinder_mask = x**2 + y**2 <= radius**2
    cylinder_array[cylinder_mask] = 1

    return cylinder_array


def create_binary_sphere(r1):
    from math import sqrt

    boxsize = 2 * r1 + 1
    kern_sphere = np.zeros(shape=(boxsize, boxsize, boxsize), dtype="float")
    kx = ky = kz = boxsize
    center = boxsize // 2
    r1 = center
    for i in range(kx):
        for j in range(ky):
            for k in range(kz):
                dist = sqrt(
                    (i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2
                )
                if dist < r1:
                    kern_sphere[i, j, k] = 1
    return kern_sphere

# Example usage with a cylinder of radius 0.2, height 1.0, and length 1.0
#radius = 0.2
#height = 1.0
#length = 1.0
#cylinder_array = create_cylinder_array(radius, height, length)

m1 = iotools.Map(name=args.mapname)
m1.read()

nx, ny, nz = m1.workarr.shape

pixsize = m1.workcell[0] / m1.workarr.shape[0]

xoffset_pix = int(args.offset_x / pixsize)
yoffset_pix = int(args.offset_y / pixsize)
zoffset_pix = int(args.offset_z / pixsize)

radius_pix = int(args.radius / pixsize)
sphere = create_binary_sphere(r1=radius_pix)
mx, my, mz = sphere.shape

arr = np.zeros(m1.workarr.shape, dtype=int)

# place the sphere at the centre of the box
dx = (nx - mx) // 2
dy = (ny - my) // 2
dz = (nz - mz) // 2
offset = 20 # pixels
arr[dx+xoffset_pix:dx+xoffset_pix+mx, 
    dy+yoffset_pix:dy+yoffset_pix+my, 
    dz+zoffset_pix:dz+zoffset_pix+mz] = sphere 

# introduce soft edges
soft_arr = make_soft(arr, 5)

# Output mrc file

m3 = iotools.Map(name='sphere.mrc')
m3.cell = m1.cell
m3.arr = soft_arr
m3.axorder = m1.axorder
m3.write()

