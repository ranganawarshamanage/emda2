# Test code to validate/find the point group of an EM map

import emda2.emda_methods2 as em
import argparse

parser = argparse.ArgumentParser(
    description='EMDA Point group validation')
parser.add_argument(
    '--half1name', type=str, required=True, help='half1 file mrc/map')
parser.add_argument(
    '--half2name', type=str, required=True, help='half2 file mrc/map')
parser.add_argument(
    '--maskname', type=str, required=True, help='mask file mrc/map')
parser.add_argument(
    '--resolution', type=float, required=True, help='resolution (A)')
args = parser.parse_args()

try:
    (
        proshade_pg,
        emda_pg,
        maskname,
        reboxedmapname
    ) = em.get_pointgroup(
        half1=args.half1name,
        half2=args.half2name,
        mask=args.maskname,
        resol=float(args.resolution)
    )
except ValueError as e:
    print(e)
