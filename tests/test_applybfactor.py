# Apply b factor to map

import numpy as np
import emda2.emda_methods2 as em
from emda2.core import iotools, plotter
import argparse, os


parser = argparse.ArgumentParser(description='Apply a B-factor on the map.')
parser.add_argument('--mapname', type=str, required=True, help='map1 file mrc/map')
parser.add_argument('--bfactors', type=float, required=True, nargs='+', help='B factors to apply')
args = parser.parse_args()


m1 = iotools.Map(name=args.mapname)
m1.read()

bfactors = args.bfactors
flist = em.apply_bfactor_to_map(
    f=np.fft.fftshift(np.fft.fftn(m1.workarr)), 
    bf_arr=bfactors, 
    uc=m1.workcell)

mapname = os.path.basename(args.mapname)

for i in range(flist.shape[3]):
    if bfactors[i] < 0.0:
        Bcode = "_blur" + str(abs(bfactors[i]))
    elif bfactors[i] > 0.0:
        Bcode = "_sharp" + str(abs(bfactors[i]))
    else:
        Bcode = "_unsharpened"
    filename_mrc = mapname[:-4] + Bcode + ".mrc"

    m2 = iotools.Map(name=filename_mrc)
    m2.cell = m1.workcell
    m2.arr = np.real(np.fft.ifftn(np.fft.ifftshift(flist[:, :, :, i])))
    m2.axorder = m1.axorder
    m2.write()

    print("Maps were written out!")
