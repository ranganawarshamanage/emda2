"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import argparse
import sys
import datetime
import emda2.config
from emda2.core import iotools, maptools, restools, plotter, fsctools, quaternions
import emda2.emda_methods2 as em


#print('EMDA COMMAND LINE OPTIONS \n')

cmdl_parser = argparse.ArgumentParser(
    prog="emda2",
    usage="%(prog)s [command] [arguments]",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

cmdl_parser.add_argument(
    "--version", 
    action="version", 
    version="%(prog)s-" + emda2.config.__version__
)
#cmdl_parser.add_argument('--help', action='help', help='show this help message and exit')

subparsers = cmdl_parser.add_subparsers(dest="command")

# find pointgroup of the map
pointg = subparsers.add_parser(
    "pointgroup",
    description="detect point group of the map"
)
pointg.add_argument("--half1", required=True,
                    type=str, help="half map 1")
pointg.add_argument("--mask", required=True,
                    type=str, help="mask file for half1")
pointg.add_argument("--resolution", required=True,
                    type=float, help="resolution of half1 (A)")

def find_pg(args):
    _ = em.get_pointgroup(
        half1=args.half1,
        mask=args.mask,
        resol=args.resolution,
    )


def main(command_line=None):
    #f = open("EMDA.txt", "w")
    #f.write("EMDA session recorded at %s.\n\n" % (datetime.datetime.now()))
    args = cmdl_parser.parse_args(command_line)
    if args.command is None:
        emda_commands()
    else:
        if args.command == 'pointgroup':
            find_pg(args)
            

def emda_commands():
    # print all possible commands in EMDA with a short description for each
    print('')
    print('USAGE:   >emda2 [option/method] [arguments]')
    print('EXAMPLE: >emda2 info --map foo.map')
    print('')
    print('arguments for each method can be obtained by running')
    print('         >emda2 option -h')
    print('e.g.     >emda2 info   -h')
    print('')
    print('EMDA COMMAND LINE OPTIONS/METHODS')
    print('---------------------------------')
    print('   pointgroup   - detect the point group symmetry of the map')
    #print('   info      - output basic information about the map')
    #print('   fsc       - computes FSC between any two maps')
    #print('   halffsc   - computes FSC between half maps')
    #print('   ccmask    - generates a mask from halfmaps')
    #print('   lowpass   - lowpass filters a map to desired resolution')
    #print('   power     - computes the rotationally average power spectrum of a map')
    #print('   resol     - estimates the resolution using half data')
    #print('   map2mtz   - coverts mrc/map to mtz file')
    #print('   mtz2map   - converts mtz file into a mrc/map file')
    #print('   resample  - resamples a mrc/map to new pixel size')
    #print('   rcc       - computes pixel-based local correlation metrics between')
    #print('               half1-half2 and fullmap-model for validation')
    #print('   mmcc      - computes map-model local correlation (does not require halfdata)')
    #print('   overlay   - superposes maps on the reference (first map on the list)')
    #print('   average   - computes likelihood average map')
    #print('   transform - apply a translation and a rotation on a map')
    #print('   bfac      - estimates a B-factor of the map')
    #print('   half2full - combines halfmaps to obtain fullmap')
    #print('   diffmap   - computes fo-fo type difference map')
    #print('   applymask - applies a mask on the map')
    #print('   scalemap  - scales one map to another map')
    #print('   bestmap   - computes the normalised and weighted map')
    #print('   predfsc   - predicts FSCs based on number of particles and B-factor')
    #print('   occ       - computes overall correlation coefficient (mean correacted)')
    #print('   mirror    - changes the handedness of map')
    #print('   model2map - computes a map from the atomic model')
    #print('   mapmask   - generates a mask from the map')
    #print('   composite - generates composite maps from several maps')
    #print('   resamplemap2map - resample one map on another map')
    #print('   magref    - estimates magnification difference of one map relative to a reference')
    #print('   com       - computes centre of mass of map')
    print('')


if __name__ == "__main__":
    main()
