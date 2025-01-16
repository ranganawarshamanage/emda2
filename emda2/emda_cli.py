"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import argparse
import re
import os
import emda2.config
from emda2.core import (
    iotools,
    # maptools,
    # restools,
    plotter,
    # fsctools,
    quaternions,
    emdalogger,
)
from numpy.fft import ifftshift, ifftn
import emda2.emda_methods2 as em


class RequiredIfMissing(argparse.Action):
    def __init__(self, option_strings, dest, required_args=None, **kwargs):
        self.required_args = required_args or []
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # Set the value of the current argument
        setattr(namespace, self.dest, values)

        # Check if any of the required arguments are provided
        missing = all(getattr(namespace, arg, None) is None for arg in self.required_args)

        if missing:
            parser.error(f"One of the arguments '{', '.join(self.required_args)}' is required when '{self.dest}' is not provided.")


# print('EMDA COMMAND LINE OPTIONS \n')

cmdl_parser = argparse.ArgumentParser(
    prog="emda2",
    usage="%(prog)s [command] [arguments]",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

cmdl_parser.add_argument(
    "--version",
    action="version",
    version="%(prog)s-" + emda2.config.__version__,
)

subparsers = cmdl_parser.add_subparsers(dest="command")

# find pointgroup of the map
pointg = subparsers.add_parser(
    "pointgroup", description="detect point group of the map"
)
pointg.add_argument("--half1", required=True, type=str, help="half map 1")
pointg.add_argument("--half2", required=True, type=str, help="half map 2")
pointg.add_argument(
    "--mask", required=False, type=str, help="mask file for half1"
)
pointg.add_argument(
    "--resolution",
    required=True,
    type=float,
    help="nominal resolution of the map (A)",
)
pointg.add_argument(
    "--resol4axref",
    required=False,
    default=5.0,
    type=float,
    help="resolution for axis refinement (5 Angs.)",
)
pointg.add_argument(
    "--outputmaps",
    action="store_true",
    help="if used, symmetry copies will be written out",
)
pointg.add_argument(
    "--symaverage",
    action="store_true",
    help="if used, symmetry averaging is carried out",
)
pointg.add_argument(
    "--axlist",
    required=False,
    default=None,
    nargs="+",
    type=float,
    help=(
        "Initial axes of [tentative] point group. Should be given as X1 Y1 Z1"
        " X2 Y2 Z2 format"
    ),
)
pointg.add_argument(
    "--orderlist",
    required=False,
    default=None,
    nargs="+",
    type=int,
    help=(
        "Initial orders of axes. Should be given as Order-of-axis1"
        " Order-of-axis2 format"
    ),
)
pointg.add_argument(
    "--fsclist",
    required=False,
    default=None,
    nargs="+",
    type=float,
    help="FSCs of axes. Should be given as FSC-of-axis1 FSC-of-axis2 format",
)
pointg.add_argument(
    "--user_pg",
    required=False,
    default=None,
    type=str,
    help="User claimed pointgroup if known",
)
pointg.add_argument(
    "--label",
    required=False,
    default=None,
    type=str,
    help="String label for the outputs",
)

# FSC between maps
calcfsc = subparsers.add_parser(
    "fsc",
    description=(
        "calculate FSC between maps. Ay number of maps can be \n"
        "given and whose FSC wrt the reference will be calculated."
    ),
)
calcfsc.add_argument(
    "--reference",
    required=True,
    type=str,
    help="reference map for FSC calculation",
)
calcfsc.add_argument(
    "--maplist",
    required=True,
    nargs="+",
    type=str,
    help="list of maps to calculate FSC",
)
calcfsc.add_argument(
    "--labels", required=False, nargs="+", type=str, help="labels for FSCs"
)

# generate density mask using halfmaps
mapmask = subparsers.add_parser(
    "mapmask", description="generate mask for protein density"
)
mapmask.add_argument("--half1", required=True, type=str, help="half map 1")
mapmask.add_argument(
    "--half2", required=False, default=None, type=str, help="half map 2"
)
mapmask.add_argument(
    "--maskname",
    required=False,
    default=None,
    type=str,
    help="maskname for output",
)

# apply transformation on map
maptransform = subparsers.add_parser(
    "transform", description="apply a transformation on a map"
)
maptransform.add_argument(
    "--map", required=True, type=str, help="input map (.map/.mrc)"
)
maptransform.add_argument(
    "--axis", required=True, nargs="+", type=float, help="rotation axis"
)
maptransform.add_argument(
    "--rotation", required=True, type=float, help="rotation in degree"
)
maptransform.add_argument(
    "--translation",
    required=False,
    default=[0.0, 0.0, 0.0],
    nargs="+",
    type=float,
    help="translation vec. in Angstrom. eg 1.0 0.0 0.0",
)
maptransform.add_argument(
    "--ibin",
    required=False,
    default=None,
    type=int,
    help="translation vec. in Angstrom. eg 1.0 0.0 0.0",
)
maptransform.add_argument(
    "--mapout",
    required=False,
    default="transformed.mrc",
    help="output map (mrc/map)",
)

# apply rotation on map by interpolating in real space
rotatemap = subparsers.add_parser(
    "rotatemap", description="apply a rotation on a map"
)
rotatemap.add_argument(
    "--map", required=True, type=str, help="input map (.map/.mrc)"
)
rotatemap.add_argument(
    "--axis", required=True, nargs="+", type=float, help="rotation axis"
)
rotatemap.add_argument(
    "--rotation", required=True, type=float, help="rotation in degree"
)
rotatemap.add_argument(
    "--mapout",
    required=False,
    default="rotatedmap_rs.mrc",
    help="output map (mrc/map)",
)

# rebox a map based on a mask
reboxmap = subparsers.add_parser(
    "reboxmap", description="apply a rotation on a map"
)
reboxmap.add_argument(
    "--map", required=True, type=str, help="input map (.map/.mrc)"
)
reboxmap.add_argument(
    "--mask", required=True, type=str, help="input mask (.map/.mrc)"
)
reboxmap.add_argument(
    "--padwidth",
    required=False,
    type=int,
    default=10,
    help="rotation in degree",
)
reboxmap.add_argument(
    "--mapout",
    required=False,
    default="emda_rbxmap.mrc",
    help="output map (mrc/map)",
)

# rebox a map based on a mask
updatecell = subparsers.add_parser(
    "updatecell", description="update the cell of the map"
)
updatecell.add_argument(
    "--map", required=True, type=str, help="input map (.map/.mrc)"
)
updatecell.add_argument(
    "--newcell",
    required=False,
    nargs="+",
    type=float,
    default=None,
    help="input newcell in the format a b c",
)
updatecell.add_argument(
    "--magf",
    required=False,
    type=float,
    default=1.0,
    help="magnification to be applied for the cell. default to 1.",
)
updatecell.add_argument(
    "--newmap",
    required=False,
    type=str,
    default="updatedmap.mrc",
    help="name for the cell updated map. default to updatedmap.mrc",
)

# modelmap calculation
model2map = subparsers.add_parser(
    "model2map", description="Simulate EM map from atomic model (PDB/mmCIF)"
)
model2map.add_argument(
    "--atomic_crdfile",
    required=True,
    type=str,
    help="Name of the atomic coordinate file (PDB/mmCIF)"
)
model2map.add_argument(
    "--resolution",
    required=True,
    type=float,
    help="Resolution of the calculated map in Angstroms"
)
model2map.add_argument(
    "--pixel_size",
    required=False,
    type=float,
    help="Desired pixel size of the calculated map in Angstroms"
)
model2map.add_argument(
    "--pad",
    required=False,
    default=10,
    type=int,
    help="Number of pixels to pad with (default=10)"
)
model2map.add_argument(
    "--shift_to_boxcenter",
    required=False,
    action="store_true",
    help="If use, the model is centered in the box before the map calculation"
)
model2map.add_argument(
    "--output_crdfile",
    required=False,
    type=str,
    default="model_for_map_calculation.cif",
    help="Name of the output coordinate file (mmCIF)"
)
model2map.add_argument(
    "--density_calculator",
    required=False,
    type=str,
    default="gemmi",
    help="Name of the density calculator (gemmi(D) | refmac)"
)
model2map.add_argument(
    "--bfactor",
    required=False,
    type=float,
    default=None,
    help="B-factor for all atoms. Default=None i.e: B-factors in the model will be used."
)
model2map.add_argument(
    "--cell",
    # action=RequiredIfMissing, 
    # required_args=['reference_map'],
    nargs="+",
    type=float,
    default=None,
    help=(
        # "Cell is required if reference_map is missing. "
        "It should be given as a list. e.g. --cell a b c alpha beta gamma"
    )
)
model2map.add_argument(
    "--dims",
    # action=RequiredIfMissing, 
    # required_args=['reference_map'],
    nargs="+",
    type=int,
    default=None,
    help=(
        # "Dimensions are required if reference_map is missing. "
        "It should be given as a list. e.g. --dims 100 100 100"
    )
)
model2map.add_argument(
    "--reference_map",
    required=False,
    type=str,
    default=None,
    help=(
        "Name of the EM map to use as the reference for cell and dims. "
        "if the reference_map is missing the cell and the dims must be "
        "specified manually.")
)
model2map.add_argument(
    "--modelmapname",
    required=False,
    type=str,
    default="emda_modelmap",
    help=(
        "Output the calculated map by this name")
)


def find_pg(args):
    # find pointgroup of the map
    _ = em.get_pointgroup(
        half1=args.half1,
        half2=args.half2,
        mask=args.mask,
        resol=args.resolution,
        resol4axref=args.resol4axref,
        symaverage=args.symaverage,
        output_maps=args.outputmaps,
        axlist=args.axlist,
        orderlist=args.orderlist,
        fsclist=args.fsclist,
        user_pg=args.user_pg,
        label=args.label,
    )


def make_mapmask(args):
    # generate density mask using halfmaps
    half1 = args.half1
    half2 = args.half2

    if args.maskname is None:
        maskname = "emdamapmask_1.mrc"
    else:
        maskname = args.maskname

    if half2 is None:
        try:
            half2 = half1.replace("half_map_1", "half_map_2")
        except NameError as e:
            print(e)
            print(
                "Please make sure half1 name includes the string _half_map_1"
            )
            raise SystemExit()

    print("Reading in %s" % half1)
    h1 = iotools.Map(half1)
    h1.read()

    print("Reading in %s" % half2)
    h2 = iotools.Map(half2)
    h2.read()

    print("Genetating masks...")
    masklist = em.mask_from_halfmaps(h1, h2)

    print("Outputting mask %s" % maskname)
    maskname1 = maskname
    mout = iotools.Map(name=maskname1)
    mout.arr = masklist[0]
    mout.cell = h1.workcell
    mout.origin = h1.origin
    mout.write()


def calc_fsc(args):
    assert len(args.maplist) > 0
    try:
        if args.labels is None:
            labels = ["map" + str(i + 1) for i in range(len(args.maplist))]
        else:
            assert len(args.labels) == len(args.maplist)
            labels = args.labels
        stmap = iotools.Map(name=args.reference)
        stmap.read()
        f1 = np.fft.fftshift(np.fft.fftn(stmap.workarr))
        nbin, res_arr, bin_idx, _ = em.get_binidx(
            stmap.workcell, stmap.workarr
        )
        fsclist = []
        for imap in args.maplist:
            mobj = iotools.Map(name=imap)
            mobj.read()
            twomapfsc = em.fsc(
                f1=f1,
                f2=np.fft.fftshift(np.fft.fftn(mobj.workarr)),
                bin_idx=bin_idx,
                nbin=nbin,
            )
            fsclist.append(twomapfsc)
        # print FSC in tabular format
        fobj = open("emda_fsc.log", "w")
        emdalogger.log_string(fobj=fobj, s=os.path.abspath(args.reference))
        for imap in args.maplist:
            emdalogger.log_string(fobj=fobj, s=os.path.abspath(imap))
        labels.insert(0, "Resol.")
        fsclist.insert(0, res_arr)
        emdalogger.log_fsc(fobj=fobj, dic=dict(zip(labels, fsclist)))
        # plot FSCs
        plotter.plot_nlines(
            res_arr=res_arr,
            list_arr=fsclist[1:],
            mapname="emda_fscs",
            curve_label=labels[1:],
            fscline=0.0,
            plot_title="FSC against referencemap",
        )
    except Exception as ex:
        print("Exception Occured!!!")
        print(ex)


def apply_transformation(args):
    # reading the map into EMDA map object
    m1 = iotools.Map(args.map)
    m1.read()
    # combining axis and rotation to generate rotmat
    rotmat = quaternions.rotmat_from_axisangle(
        axis=args.axis, theta=np.deg2rad(args.rotation)
    )
    frt, newcell = em.apply_transformation(
        m1=m1, rotmat=rotmat, trans=args.translation, ibin=args.ibin
    )
    m2 = iotools.Map(name=args.mapout)
    m2.arr = np.real(ifftshift(ifftn(ifftshift(frt))))
    m2.cell = newcell  # m1.workcell
    m2.origin = m1.origin
    m2.write()


def rotate_map(args):
    # reading the map into EMDA map object
    m1 = iotools.Map(args.map)
    m1.read()
    # combining axis and rotation to generate rotmat
    rotmat = quaternions.rotmat_from_axisangle(
        axis=args.axis, theta=np.deg2rad(args.rotation)
    )
    rho = em.rotate_map_realspace(
        m1=m1,
        rotmat=rotmat,
    )
    # output rotated map
    m2 = iotools.Map(name=args.mapout)
    m2.arr = rho
    m2.cell = m1.workcell
    m2.origin = m1.origin
    m2.write()


def rebox_map(args):
    m1 = iotools.Map(args.map)
    m1.read()
    mm = iotools.Map(args.mask)
    mm.read()
    reboxed_map, reboxed_mask = em.rebox_by_mask(
        arr=m1.workarr,
        mask=mm.workarr,
        mask_origin=mm.origin,
        padwidth=args.padwidth,
    )
    # output reboxed maps
    pix = m1.workcell[0] / m1.workarr.shape[0]
    newcell = [sh * pix for i, sh in enumerate(reboxed_map.shape)]
    for _ in range(3):
        newcell.append(90.0)
    mrbx = iotools.Map(args.mapout)
    mrbx.arr = reboxed_map
    mrbx.cell = newcell
    mrbx.origin = [0, 0, 0]
    mrbx.write()
    mmrbx = iotools.Map("emda_rbxmask.mrc")
    mmrbx.arr = reboxed_mask
    mmrbx.cell = newcell
    mmrbx.origin = [0, 0, 0]
    mmrbx.write()


def update_cell(args):
    m1 = iotools.Map(args.map)
    m1.read()
    m2 = iotools.Map(args.newmap)
    m2.cell = m1.cell
    m2.arr = m1.arr
    m2.origin = m1.origin
    if args.newcell is not None:
        args.magf = None
        if len(args.newcell) > 3:
            m2.cell = args.newcell[:3]
    elif args.magf is not None:
        m2.cell = [a * args.magf for a in m1.cell]
    else:
        print("No change in cell!")
    m2.write()


def model2map(args):
    input_crdfile = args.atomic_crdfile
    angpix = args.pixel_size
    pad = args.pad
    shift_to_boxcenter = args.shift_to_boxcenter
    output_crdfile = args.output_crdfile
    density_calculator = args.density_calculator
    bfactor = args.bfactor
    cell = args.cell
    dims = args.dims
    output_density_file = args.modelmapname
    reference_map = args.reference_map
    resolution = args.resolution

    if reference_map:
        mref = iotools.Map(reference_map)
        mref.read()

        angpix = min([mref.workcell[i] / mref.workarr.shape[i] for i in range(3)])
        warnings.warn(f"Provided pixel size will be overwritten with {angpix} A/pix")
        cell = mref.workcell
        dims = mref.workarr.shape

    if cell and dims:
        angpix = cell[0] / dims[0]

    if angpix is None:
        angpix = resolution / 2

    print("===== INPUT PARAMETERS =====")
    # Printing parameters
    print(f"Input coordinate file: {input_crdfile}")
    print(f"Resolution: {resolution}")
    print(f"Pixel size: {angpix}")
    print(f"Shift the model to box center: {shift_to_boxcenter}")
    print(f"Pad: {pad}")
    print(f"Cell: {cell}")
    if dims is None:
        print("Dimensions are not given. They will be calculated automatically")
    else:
        print(f"Dimensions: {dims}")
    print(f"Density calculator: {density_calculator}")
    if bfactor is None:
        print("Atomic B-factors will be used.")
    else:
        print(f"Set all atomic Bfactors to: {bfactor}")
    if reference_map:
        print(f"Reference map: {reference_map}")
    print(f"Output coordinates file: {output_crdfile}")
    print(f"Output calculated map: {output_density_file}.mrc")
    print("============================")

    modelmap, cell = em.model2map(
        input_crdfile=input_crdfile, resolution=resolution, pad=pad, 
        shift_to_boxcenter=shift_to_boxcenter, 
        output_crdfile=output_crdfile, 
        density_calculator=density_calculator, 
        bfactor=bfactor, cell=cell, dims=dims, angpix=angpix)

    mout = iotools.Map(f"{output_density_file}.mrc")
    mout.cell = cell
    mout.arr = modelmap
    mout.write()

    print("Calculated map was successfully written.")



def main(command_line=None):
    # f = open("EMDA.txt", "w")
    # f.write("EMDA session recorded at %s.\n\n" % (datetime.datetime.now()))
    args = cmdl_parser.parse_args(command_line)
    if args.command is None:
        emda_commands()
    else:
        if args.command == "pointgroup":
            find_pg(args)
        if args.command == "mapmask":
            make_mapmask(args)
        if args.command == "fsc":
            calc_fsc(args)
        if args.command == "transform":
            apply_transformation(args)
        if args.command == "rotatemap":
            rotate_map(args)
        if args.command == "reboxmap":
            rebox_map(args)
        if args.command == "updatecell":
            update_cell(args)
        if args.command == "model2map":
            model2map(args)


def emda_commands():
    # print all possible commands in EMDA with a short description for each
    print("")
    print("USAGE:   >emda2 [option/method] [arguments]")
    print("EXAMPLE: >emda2 info --map foo.map")
    print("")
    print("arguments for each method can be obtained by running")
    print("         >emda2 option -h")
    print("e.g.     >emda2 info   -h")
    print("")
    print("EMDA COMMAND LINE OPTIONS/METHODS")
    print("---------------------------------")
    print("   pointgroup   - detect the point group symmetry of the map")
    print("   fsc          - computes FSC between two maps")
    print("   mapmask      - generates a mask using halfmaps")
    print(
        "   transform    - apply a transformation on the map (Fourier space)"
    )
    print("   rotatemap    - apply a rotation on the map (Real space)")
    print("   rebox        - rebox a map based on a mask")
    print("   updatecell   - update the cell")
    print('   model2map    - computes a map from the atomic model')
    # print('   info      - output basic information about the map')

    # print('   halffsc   - computes FSC between half maps')
    # print('   ccmask    - generates a mask from halfmaps')
    # print('   lowpass   - lowpass filters a map to desired resolution')
    # print('   power     - computes the rotationally average power spectrum of a map')
    # print('   resol     - estimates the resolution using half data')
    # print('   map2mtz   - coverts mrc/map to mtz file')
    # print('   mtz2map   - converts mtz file into a mrc/map file')
    # print('   resample  - resamples a mrc/map to new pixel size')
    # print('   rcc       - computes pixel-based local correlation metrics between')
    # print('               half1-half2 and fullmap-model for validation')
    # print('   mmcc      - computes map-model local correlation (does not require halfdata)')
    # print('   overlay   - superposes maps on the reference (first map on the list)')
    # print('   average   - computes likelihood average map')
    # print('   transform - apply a translation and a rotation on a map')
    # print('   bfac      - estimates a B-factor of the map')
    # print('   half2full - combines halfmaps to obtain fullmap')
    # print('   diffmap   - computes fo-fo type difference map')
    # print('   applymask - applies a mask on the map')
    # print('   scalemap  - scales one map to another map')
    # print('   bestmap   - computes the normalised and weighted map')
    # print('   predfsc   - predicts FSCs based on number of particles and B-factor')
    # print('   occ       - computes overall correlation coefficient (mean correacted)')
    # print('   mirror    - changes the handedness of map')
    
    # print('   mapmask   - generates a mask from the map')
    # print('   composite - generates composite maps from several maps')
    # print('   resamplemap2map - resample one map on another map')
    # print('   magref    - estimates magnification difference of one map relative to a reference')
    # print('   com       - computes centre of mass of map')
    print("")


if __name__ == "__main__":
    main()
