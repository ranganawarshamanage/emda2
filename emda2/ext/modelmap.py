import os
import emda2.emda_methods2 as em
from emda2.core import iotools, modeltools
import numpy as np
import gemmi as gemmi
import argparse

# parser = argparse.ArgumentParser(description='Simulate EM map from an atomic model')
# parser.add_argument('--atomic_model', type=str, required=True, help='Input PDB/mmCIF')
# parser.add_argument('--resolution', type=float, required=True, help='Specify the resolution in Angstroms')
# parser.add_argument('--em_mapname', type=str, required=False, 
#                     help='If the map is given the cell and the dims will be taken from this map')
# parser.add_argument('--global_B', type=float, required=False, default=20., help='Set global B factor (default=20).')
# parser.add_argument('--output_mapname', type=str, required=False, default='emda_modelmap.mrc', help='Output map name.')
# args = parser.parse_args()



#def pdb_center_and_setb_2_map(pdbfile, mapfile, angpix, pad=0, bfactor=-1):
#    st = gemmi.read_structure(pdbfile)
#    st.add_entity_types()
#    low_bounds = [float("+inf")] * 3
#    high_bounds = [float("-inf")] * 3
#    delta = [float(0)] * 3
#    for chain in st[0]:
#        polymer = chain.get_polymer()
#        if polymer:
#            for residue in polymer:
#                for atom in residue:
#                    # set bfactor
#                    if bfactor >= 0.0:
#                        atom.b_iso = bfactor
#                    for i in range(3):
#                        if atom.pos[i] < low_bounds[i]:
#                            low_bounds[i] = atom.pos[i]
#                        if atom.pos[i] > high_bounds[i]:
#                            high_bounds[i] = atom.pos[i]
#
#    for i in range(3):
#        delta[i] = high_bounds[i] - low_bounds[i]
#
#    # Now have all coordinates start at zero
#    for chain in st[0]:
#        polymer = chain.get_polymer()
#        if polymer:
#            for residue in polymer:
#                for atom in residue:
#                    for i in range(3):
#                        atom.pos[i] = pad + atom.pos[i] - low_bounds[i]
#
#    cenpdbfile = "cen_" + pdbfile
#    st.write_pdb(cenpdbfile)
#    mymaxdim = np.max(delta)
#    mycell = np.ceil(mymaxdim) + 2 * pad
#    cell = [mycell, mycell, mycell]
#    min_dims = np.asarray(cell, dtype="float") / angpix
#    min_dims = np.ceil(min_dims).astype(int)
#    modelmap = em.model2map(
#        modelxyz=cenpdbfile, dim=min_dims, resol=2 * angpix, cell=cell
#    )
#    write_mrc(modelmap, mapfile, cell)
#    print("Done! written ", mapfile, " with cell= ", cell)
#
#
#def calculate_map(pdbfile, angpix, mapfile="emda_modelmap.mrc", pad=0, bfactor=20):
#    if pdbfile.endswith(".pdb"):
#        modeltools.pdb2mmcif(pdbfile)
#
#    doc = gemmi.cif.read_file('out.cif')
#    block = doc.sole_block()
#    col_x = block.find_values("_atom_site.Cartn_x")
#    col_y = block.find_values("_atom_site.Cartn_y")
#    col_z = block.find_values("_atom_site.Cartn_z")
#    b_iso = block.find_values("_atom_site.B_iso_or_equiv")
#
#    min_x = np.amin(np.array(col_x, dtype='float'))
#    min_y = np.amin(np.array(col_y, dtype='float'))
#    min_z = np.amin(np.array(col_z, dtype='float'))
#    max_x = np.amax(np.array(col_x, dtype='float'))
#    max_y = np.amax(np.array(col_y, dtype='float'))
#    max_z = np.amax(np.array(col_z, dtype='float'))
#
#    low_bound = np.array([min_x, min_y, min_z], dtype='float')
#    high_bound = np.array([max_x, max_y, max_z], dtype='float')
#    delta = high_bound - low_bound
#
#    mymaxdim = np.max(delta)
#    mycell = np.ceil(mymaxdim) + 2 * pad
#
#    for n, _ in enumerate(col_x):
#        # set atomic B factor
#        b_iso[n] = str(bfactor)
#
#        # place the model at (0,0,0)
#        col_x[n] = str(float(col_x[n]) - low_bound[0])
#        col_y[n] = str(float(col_y[n]) - low_bound[1])
#        col_z[n] = str(float(col_z[n]) - low_bound[2])
#
#    # shifts to center of the cell
#    shifts = [(mycell - x) / 2 for x in delta]
#    # place the model at the center of the cell
#    for n, _ in enumerate(col_x):
#        col_x[n] = str(float(col_x[n]) + shifts[0])
#        col_y[n] = str(float(col_y[n]) + shifts[1])
#        col_z[n] = str(float(col_z[n]) + shifts[2])
#
#    # write out the centered model
#    doc.write_file('cenmodel.cif')
#
#    cell = np.array([mycell, mycell, mycell], "float")
#    min_dims = np.ceil(cell / angpix).astype(int)
#    modelmap = model2map_gm(
#        modelxyz='cenmodel.cif', dim=min_dims, resol=2 * angpix, cell=cell
#    )
#    m1 = iotools.Map(mapfile)
#    m1.arr = modelmap
#    m1.cell = cell
#    m1.write()
#    print("Done! written ", mapfile, " with cell= ", cell)
#
#
#def model2map_refmac(
#    modelxyz, dim, resol, cell, bfac=None, maporigin=None, ligfile=None, outputpath=None, shift_to_boxcenter=False,
#):
#    """Calculates EM map from atomic coordinates using REFMAC5
#
#    Args:
#        modelxyz (string): Name of the coordinate file (.cif/.pdb)
#        dim (list): Map dimensions [nx, ny, nz] as a list of integers
#        resol (float): Requested resolution for density calculation in Angstroms.
#        cell (list): Cell parameters a, b and c as floats
#        maporigin (list, optional): Location of the first column (nxstart), 
#            row (nystart), section (nzstart) of the unit cell. Defaults to [0, 0, 0].
#        ligfile (string, optional): Name of the ligand description file. Defaults to None.
#        outputpath (string, optional): Path for auxilliary files. Defaults to current
#            working directory.
#        bfac(float, optional): Parameter for refmac. Set all atomic B values to bfac
#            when it is positive. Default to None.
#        shift_to_boxcenter (bool, optional): This parameter is useful if the calculated
#            map from the model needs to be placed at the center of the box. Default to
#            False. Also, the shifted model will be written to outputpath directory.
#
#    Returns:
#        float ndarray: calculated model-based density array
#    """
#    import gemmi as gm
#    import shutil
#    import os
#    from emda2.core import mtz
#
#    # print parameters
#    print('Requested resolution (A): ', resol)
#    print('Requested sampling: ', dim)
#    print('Cell [a, b, c]: ', cell)
#    if outputpath is None:
#        outputpath = os.getcwd()
#    outputpath = os.path.join(outputpath, 'emda_refmacfiles/')
#    print('outputpath: ', outputpath)
#    # make director for files for refmac run
#    if os.path.exists(outputpath):
#        shutil.rmtree(outputpath)
#    os.mkdir(outputpath) 
#    # check for valid sampling:
#    for i in range(3):
#        if dim[i] % 2 != 0:
#            dim[i] += 1
#    # check for minimum sampling
#    min_pix_size = resol / 2  # in Angstrom
#    min_dim = np.asarray(cell[:3], dtype="float") / min_pix_size
#    min_dim = np.ceil(min_dim).astype(int)
#    for i in range(3):
#        if min_dim[i] % 2 != 0:
#            min_dim += 1
#        if min_dim[0] > dim[0]:
#            print("Requested dims: ", dim)
#            print("Minimum dims needed (for requested resolution): ", min_dim)
#            print("!!! Please lower the requested resolution or increase the grid dimensions !!!")
#            raise SystemExit()
#    # replace/add cell and write model.cif
#    if shift_to_boxcenter:
#        from emda.core.modeltools import shift_to_origin,shift_model
#        doc = shift_to_origin(modelxyz)
#        doc.write_file(outputpath+"model1.cif")
#        modelxyz = outputpath+"model1.cif"
#    # run refmac using model.cif just created
#    a, b, c = cell[:3]
#    structure = gm.read_structure(modelxyz)
#    structure.cell.set(a, b, c, 90.0, 90.0, 90.0)
#    structure.spacegroup_hm = "P 1"
#    structure.make_mmcif_document().write_file(outputpath+"model.cif")
#    # run refmac using model.cif just created
#    run_refmac_sfcalc(filename=outputpath+"model.cif", 
#                              resol=resol, 
#                              ligfile=ligfile,
#                              bfac=bfac)
#    modelmap, _ = mtz.mtz2map(outputpath+"sfcalc_from_crd.mtz", dim)
#    if shift_to_boxcenter:
#        maporigin = None # no origin shift allowed
#        modelmap = np.fft.fftshift(modelmap) #bring modelmap to boxcenter
#        # shift model to boxcenter
#        doc = shift_model(mmcif_file=outputpath+"model.cif", shift=[a/2, b/2, c/2])
#        doc.write_file(outputpath+"emda_shifted_model.cif")
#    if maporigin is None:
#        maporigin = [0, 0, 0]
#    else:
#        shift_z = maporigin[0]
#        shift_y = maporigin[1]
#        shift_x = maporigin[2]
#        modelmap = np.roll(
#            np.roll(np.roll(modelmap, -shift_z, axis=0), -shift_y, axis=1),
#            -shift_x,
#            axis=2,
#        )
#    return modelmap
#
#
#def run_refmac_sfcalc(filename, resol, lig=True, bfac=None, ligfile=None):
#    import os
#    import os.path
#    import subprocess
#
#    #
#    current_path = os.getcwd()  # get current path
#    filepath = os.path.abspath(os.path.dirname(filename)) + "/"
#    os.chdir(filepath)
#    fmtz = filename[:-4] + ".mtz"
#    cmd = ["refmac5", "XYZIN", filename, "HKLOUT", fmtz]
#    if ligfile is not None:
#        cmd = ["refmac5", "XYZIN", filename, "HKLOUT", fmtz, "lib_in", ligfile]
#        lig = False
#    # Creating the sfcalc.inp with custom parameters (resol, Bfac)
#    sfcalc_inp = open(filepath + "sfcalc.inp", "w+")
#    sfcalc_inp.write("mode sfcalc\n")
#    sfcalc_inp.write("sfcalc cr2f\n")
#    if lig:
#        sfcalc_inp.write("make newligand continue\n")
#    sfcalc_inp.write("resolution %f\n" % resol)
#    if bfac is not None and bfac > 0.0:
#        sfcalc_inp.write("temp set %f\n" % bfac)
#    sfcalc_inp.write("source em mb\n")
#    sfcalc_inp.write("make hydrogen yes\n")
#    sfcalc_inp.write("end")
#    sfcalc_inp.close()
#    # Read in sfcalc_inp
#    PATH = filepath + "sfcalc.inp"
#    logf = open(filepath + "sfcalc.log", "w+")
#    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
#        print("sfcalc.inp exists and is readable")
#        inp = open(filepath + "sfcalc.inp", "r")
#        # Run the command with parameters from file f2mtz.inp
#        subprocess.call(cmd, stdin=inp, stdout=logf)
#        logf.close()
#        inp.close()
#    else:
#        raise SystemExit("File is either missing or not readable")
#    os.chdir(current_path)
#

def run_refmac_sfcalc2(filename, resol, dim, lig=True, bfac=None, ligfile=None):
    import os
    import os.path
    import subprocess
    from emda2.core import mtz

    filename = os.path.abspath(filename)

    # Print input parameters
    print('Coordinate filename:', filename)
    print('Resolution: ', resol)

    # Command for REFMAC
    fmtz = filename[:-4] + ".mtz"
    cmd = ["refmac5", "XYZIN", filename, "HKLOUT", fmtz]
    if ligfile is not None:
        cmd = ["refmac5", "XYZIN", filename, "HKLOUT", fmtz, "lib_in", ligfile]
        lig = False

    # Creating the sfcalc.inp
    sfcalc_inp = open("sfcalc.inp", "w")
    sfcalc_inp.write("mode sfcalc\n")
    sfcalc_inp.write("sfcalc cr2f\n")
    if lig:
        sfcalc_inp.write("make newligand continue\n")
    sfcalc_inp.write("resolution %f\n" % resol)
    if bfac is not None and bfac > 0.0:
        sfcalc_inp.write("temp set %f\n" % bfac)
    sfcalc_inp.write("source em mb\n")
    sfcalc_inp.write("make hydrogen yes\n")
    sfcalc_inp.write("end")
    sfcalc_inp.close()
    # Read in sfcalc_inp
    PATH = "sfcalc.inp"
    logf = open("sfcalc.log", "w")
    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print("sfcalc.inp exists and is readable")
        inp = open("sfcalc.inp", "r")
        # Run the command with parameters from file f2mtz.inp
        subprocess.call(cmd, stdin=inp, stdout=logf)
        logf.close()
        inp.close()
    else:
        raise SystemExit("File is either missing or not readable")
    
    if os.path.isfile("sfcalc_from_crd.mtz"):
        modelmap, _ = mtz.mtz2map("sfcalc_from_crd.mtz", dim)
        return modelmap.transpose()  # Transpose bcoz F --> C
    else:
        raise SystemExit("sfcalc_from_crd.mtz is either missing or not readable")
    

def model2map_gm(st, resol, dim):
    import gemmi, shutil, os

    from servalcat.utils.model import calc_fc_fft

    asu_data = calc_fc_fft(st=st, 
                           d_min=resol, 
                           source='electron', 
                           mott_bethe=True)
    griddata = asu_data.get_f_phi_on_grid(dim)
    griddata_np = (np.array(griddata, copy=False))
    modelmap = (np.fft.ifftn(np.conjugate(griddata_np))).real

    """ 
    if maporigin is None:
        maporigin = [0, 0, 0]
    else:
        shift_z = maporigin[0]
        shift_y = maporigin[1]
        shift_x = maporigin[2]
        # print(shift_z, shift_y, shift_x)
        modelmap = np.roll(
            np.roll(np.roll(modelmap, -shift_z, axis=0), -shift_y, axis=1),
            -shift_x,
            axis=2,
        ) """
    return modelmap


def density_calculation(
        input_crdfile, resolution, pad=0, shift_to_boxcenter=False, 
        output_crdfile='model_for_map_calculation.cif',
        compute_map_with='gemmi',bfactor=None, cell=None, dims=None,
        angpix=None
):
    if angpix is None:
        angpix = resolution / 2

    if cell is None:
        shift_to_boxcenter = True

    if input_crdfile.endswith(".pdb"):
        modeltools.pdb2mmcif(input_crdfile)
    try:
        doc = gemmi.cif.read_file(input_crdfile)
    except:
        doc = gemmi.cif.read_file('out.cif')

    block = doc.sole_block()
    col_x = block.find_values("_atom_site.Cartn_x")
    col_y = block.find_values("_atom_site.Cartn_y")
    col_z = block.find_values("_atom_site.Cartn_z")
    b_iso = block.find_values("_atom_site.B_iso_or_equiv")

    min_x = np.amin(np.array(col_x, dtype='float'))
    min_y = np.amin(np.array(col_y, dtype='float'))
    min_z = np.amin(np.array(col_z, dtype='float'))
    max_x = np.amax(np.array(col_x, dtype='float'))
    max_y = np.amax(np.array(col_y, dtype='float'))
    max_z = np.amax(np.array(col_z, dtype='float'))

    low_bound = np.array([min_x, min_y, min_z], dtype='float')
    high_bound = np.array([max_x, max_y, max_z], dtype='float')
    delta = high_bound - low_bound

    # Minimum cell
    mymaxdim = np.max(delta)
    mycell = np.ceil(mymaxdim) + 2 * pad
    # Make cell
    min_cell = np.array([mycell, mycell, mycell], "float")    

    if cell is not None:
        # Make sure given cell is bigger than min_cell
        if np.amin(np.array([cell[i] - min_cell[i] for i in range(3)], "float")) < 0.:
            raise SystemExit(f"given cell {cell} vs required minimum cell {min_cell}")
        cell = np.array(cell, dtype=float)
        # shifts to center of the cell
        shifts = [(cell[i] - delta[i]) / 2 for i in range(3)]
    else:
        cell = min_cell
        # shifts to center of the cell
        shifts = [(mycell - x) / 2 for x in delta]

    # print(f"cell {cell}")
    # print(f"delta {delta}")
    # print(f"shifts {shifts}")

    # Calculate min_dims from cell and pixel size
    min_dims = np.ceil(np.round(cell[:3] / angpix, decimals=3)).astype(int)
    print(f"cell/angpix: {np.round(cell[:3] / angpix, decimals=3)}")
    print(f"min_dims: {min_dims}")
    min_dims = min_dims + min_dims % 2

    if dims is not None:
        dims = np.asarray(dims, dtype=int)
        dims = dims + dims % 2

        if np.amin(dims - min_dims) < 0:
            print(f"Problem! requested dims {dims} vs. required dims {min_dims}")
            print(f"Warning! grid was changed to {min_dims}")
            dims = min_dims
    else:
        print(f"Warning! grid is set to {min_dims}")
        dims = min_dims

    if bfactor is not None:
        for n, _ in enumerate(col_x):
            # set atomic B factor
            b_iso[n] = str(bfactor)

    if shift_to_boxcenter:
        # first place the model at (0,0,0), then shift to center of the cell
        for n, _ in enumerate(col_x):
            col_x[n] = str((float(col_x[n]) - low_bound[0]) + shifts[0])
            col_y[n] = str((float(col_y[n]) - low_bound[1]) + shifts[1])
            col_z[n] = str((float(col_z[n]) - low_bound[2]) + shifts[2])

    st = gemmi.make_structure_from_block(doc[0])
    a, b, c = cell[:3]
    st.spacegroup_hm = "P 1"
    st.cell.set(a, b, c, 90., 90., 90.)
    st.make_mmcif_document().write_file(output_crdfile)
    print(f"Coordinate file for map calculation is {output_crdfile}")

    if resolution < 2 * angpix:
        print(f"Problem! Requested resolution {resolution} A vs Nyquist resolution {2 * angpix} A")
        print(f"Warning! Resolution was changed to Nyquist resolution {2 * angpix} A")
        resolution = 2 * angpix

    if compute_map_with in ['refmac', 'refmac5', 'REFMAC', 'REFMAC5']:
        # REFMAC map calculation
        modelmap = run_refmac_sfcalc2(output_crdfile, resol=resolution, dim=dims)
    else:
        # Gemmi map calculation
        modelmap = model2map_gm(st=st, dim=dims, resol=resolution)

    return [modelmap, cell]


# if __name__ == "__main__":
# 
#     """ pdbfile = "8ge9.pdb"
#     mapfile = "emda_modelmap.mrc"
#     angpix = 2.0
#     bfactor = 20.0
#     pad = 10
#     # pdb_center_and_setb_2_map(pdbfile, mapfile, angpix, pad=pad, bfactor=-1)
#     calculate_map(pdbfile, angpix, mapfile, pad=pad, bfactor=bfactor) """
# 
# 
#     modelxyz = args.atomic_model
#     resol = args.resolution
#     outputmap = args.output_mapname
#     Bfactor = args.global_B
# 
#     m1 = iotools.Map(args.em_mapname)
#     m1.read()
# 
#     dim = m1.workarr.shape
#     cell = m1.workcell
# 
#     modelmap, newcell = density_calculation(
#         input_crdfile=modelxyz,
#         angpix=resol / 2,
#         cell=cell,
#         dims=dim,
#         bfactor=30,
#         output_crdfile='test_model2.cif',
#         shift_to_boxcenter=False,
#         pad=10,
#         resolution=resol,
#     )
# 
#     # Output map
#     m1 = iotools.Map(outputmap)
#     m1.arr = modelmap
#     m1.cell = newcell
#     m1.write()
#     print("Done! written ", outputmap, " with cell= ", newcell)