# Simulate map fron atomic model

import numpy as np
import emda2.emda_methods2 as em
from emda2.core import iotools, maptools
import argparse, os


parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--mapname', type=str, required=True, help='map file mrc/map')
parser.add_argument(
    '--atomic_model', type=str, required=True, help='pdb/mmcif file mrc/map')
parser.add_argument(
    '--resolution', type=float, required=True, help='Specify resolution')
parser.add_argument(
    '--outputmapname', type=str, default=None, required=False, help='outputmap name mrc/map')
args = parser.parse_args()


def model2map_gm(
    modelxyz, 
    resol, 
    dim, 
    cell, 
    maporigin=None, 
    outputpath=None, 
    shift_to_boxcenter=False
):
    import gemmi, shutil
    from servalcat.utils.model import calc_fc_fft

    if outputpath is None:
        outputpath = os.getcwd()
    outputpath = os.path.join(outputpath, 'emda_gemmifiles/')
    print('outputpath: ', outputpath)
    # make director for files for refmac run
    if os.path.exists(outputpath):
        shutil.rmtree(outputpath)
    os.mkdir(outputpath) 
    # check for valid sampling:
    for i in range(3):
        if dim[i] % 2 != 0:
            dim[i] += 1
    # check for minimum sampling
    min_pix_size = resol / 2  # in Angstrom
    min_dim = np.asarray(cell[:3], dtype="float") / min_pix_size
    min_dim = np.ceil(min_dim).astype(int)
    for i in range(3):
        if min_dim[i] % 2 != 0:
            min_dim += 1
        if min_dim[0] > dim[0]:
            print("Requested dims: ", dim)
            print("Minimum dims needed (for requested resolution): ", min_dim)
            print("!!! Please lower the requested resolution or increase the grid dimensions !!!")
            raise SystemExit()
    # if shift_to_boxcenter:
    #     from emda.core.modeltools import shift_to_origin,shift_model
    #     doc = shift_to_origin(modelxyz)
    #     doc.write_file(outputpath+"model1.cif")
    #     modelxyz = outputpath+"model1.cif"
    a, b, c = cell[:3]
    st = gemmi.read_structure(modelxyz)
    st.spacegroup_hm = "P 1"
    st.cell.set(a, b, c, 90., 90., 90.)
    st.make_mmcif_document().write_file(outputpath+"model.cif")
    asu_data = calc_fc_fft(st=st, 
                           d_min=resol, 
                           source='electron', 
                           mott_bethe=True)
    griddata = asu_data.get_f_phi_on_grid(dim)
    griddata_np = (np.array(griddata, copy=False)).transpose()
    modelmap = (np.fft.ifftn(np.conjugate(griddata_np))).real
    if np.sum(np.asarray(modelmap.shape, 'int') - np.asarray(dim, 'int')) != 0:
        cpix = [cell[i]/shape for i, shape in enumerate(modelmap.shape)]
        tpix = [cell[i]/shape for i, shape in enumerate(dim)]
        modelmap = em.resample_data(
            curnt_pix=cpix, targt_pix=tpix, arr=modelmap, targt_dim=dim)
    # if shift_to_boxcenter:
    #     maporigin = None # no origin shift allowed
    #     modelmap = np.fft.fftshift(modelmap) #bring modelmap to boxcenter
    #     # shift model to boxcenter
    #     doc = shift_model(mmcif_file=outputpath+"model.cif", shift=[a/2, b/2, c/2])
    #     doc.write_file(outputpath+"emda_shifted_model.cif")
    if maporigin is None:
        maporigin = [0, 0, 0]
    else:
        shift_z = maporigin[0]
        shift_y = maporigin[1]
        shift_x = maporigin[2]
        modelmap = np.roll(
            np.roll(np.roll(modelmap, -shift_z, axis=0), -shift_y, axis=1),
            -shift_x,
            axis=2,
        )
    return modelmap


def model2map_refmac(
    modelxyz, dim, resol, cell, bfac=None, maporigin=None, ligfile=None, outputpath=None, shift_to_boxcenter=False,
):
    """Calculates EM map from atomic coordinates using REFMAC5

    Args:
        modelxyz (string): Name of the coordinate file (.cif/.pdb)
        dim (list): Map dimensions [nx, ny, nz] as a list of integers
        resol (float): Requested resolution for density calculation in Angstroms.
        cell (list): Cell parameters a, b and c as floats
        maporigin (list, optional): Location of the first column (nxstart), 
            row (nystart), section (nzstart) of the unit cell. Defaults to [0, 0, 0].
        ligfile (string, optional): Name of the ligand description file. Defaults to None.
        outputpath (string, optional): Path for auxilliary files. Defaults to current
            working directory.
        bfac(float, optional): Parameter for refmac. Set all atomic B values to bfac
            when it is positive. Default to None.
        shift_to_boxcenter (bool, optional): This parameter is useful if the calculated
            map from the model needs to be placed at the center of the box. Default to
            False. Also, the shifted model will be written to outputpath directory.

    Returns:
        float ndarray: calculated model-based density array
    """
    import gemmi as gm
    import shutil

    # print parameters
    print('Requested resolution (A): ', resol)
    print('Requested sampling: ', dim)
    print('Cell [a, b, c]: ', cell)
    if outputpath is None:
        outputpath = os.getcwd()
    outputpath = os.path.join(outputpath, 'emda_refmacfiles/')
    print('outputpath: ', outputpath)
    # make director for files for refmac run
    if os.path.exists(outputpath):
        shutil.rmtree(outputpath)
    os.mkdir(outputpath) 
    # check for valid sampling:
    for i in range(3):
        if dim[i] % 2 != 0:
            dim[i] += 1
    # check for minimum sampling
    min_pix_size = resol / 2  # in Angstrom
    min_dim = np.asarray(cell[:3], dtype="float") / min_pix_size
    min_dim = np.ceil(min_dim).astype(int)
    for i in range(3):
        if min_dim[i] % 2 != 0:
            min_dim += 1
        if min_dim[0] > dim[0]:
            print("Requested dims: ", dim)
            print("Minimum dims needed (for requested resolution): ", min_dim)
            print("!!! Please lower the requested resolution or increase the grid dimensions !!!")
            raise SystemExit()
    # replace/add cell and write model.cif
    # if shift_to_boxcenter:
    #     from emda.core.modeltools import shift_to_origin,shift_model
    #     doc = shift_to_origin(modelxyz)
    #     doc.write_file(outputpath+"model1.cif")
    #     modelxyz = outputpath+"model1.cif"
    # run refmac using model.cif just created
    a, b, c = cell[:3]
    structure = gm.read_structure(modelxyz)
    structure.cell.set(a, b, c, 90.0, 90.0, 90.0)
    structure.spacegroup_hm = "P 1"
    structure.make_mmcif_document().write_file(outputpath+"model.cif")
    # run refmac using model.cif just created
    iotools.run_refmac_sfcalc(filename=outputpath+"model.cif", 
                              resol=resol, 
                              ligfile=ligfile,
                              bfac=bfac)
    modelmap, uc = maptools.mtz2map(outputpath+"sfcalc_from_crd.mtz", dim)
    print(modelmap.shape)
    # if shift_to_boxcenter:
    #     maporigin = None # no origin shift allowed
    #     modelmap = np.fft.fftshift(modelmap) #bring modelmap to boxcenter
    #     # shift model to boxcenter
    #     doc = shift_model(mmcif_file=outputpath+"model.cif", shift=[a/2, b/2, c/2])
    #     doc.write_file(outputpath+"emda_shifted_model.cif")
    """ if maporigin is None:
        maporigin = [0, 0, 0]
    else:
        shift_z = maporigin[0]
        shift_y = maporigin[1]
        shift_x = maporigin[2]
        modelmap = np.roll(
            np.roll(np.roll(modelmap, -shift_z, axis=0), -shift_y, axis=1),
            -shift_x,
            axis=2,
        ) """
    return modelmap


if __name__ == "__main__":

    m1 = iotools.Map(name=args.mapname)
    m1.read()

    #   
    """ modelmap = model2map_gm(
        modelxyz=args.atomic_model, 
        resol=args.resolution,
        dim=m1.workarr.shape, 
        cell=m1.workcell, 
        maporigin=m1.origin,
        # shift_to_boxcenter=args.shift_to_boxcenter,
        ) """

    # REFMAC sfcalc
    modelmap = model2map_refmac(
        modelxyz=args.atomic_model,
        dim=[272, 272, 272], #  m1.workarr.shape,
        resol=args.resolution,
        cell=m1.workcell,
        maporigin=m1.origin,
        # lig=args.lig,
        # ligfile=args.lgf,
        # shift_to_boxcenter=args.shift_to_boxcenter,
    )


if args.outputmapname is None:
    modelmapname = "emda_modelmap.mrc"
else:
    modelmapname = args.outputmapname

m2 = iotools.Map(name=modelmapname)
# croppedImage = iotools.cropimage(arr=modelmap, tdim=m1.arr.shape)
m2.cell = m1.workcell
m2.arr = modelmap
m2.axorder = m1.axorder
m2.write()
