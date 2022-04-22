import gemmi, os
import numpy as np


def pdb2mmcif(filename_pdb):
    structure = gemmi.read_structure(filename_pdb)
    structure.setup_entities()
    structure.assign_label_seq_id()
    mmcif = structure.make_mmcif_document()
    mmcif.write_file("out.cif")


def model_rebox(mask, mmcif_file, padwidth=10, uc=None):
    mask = mask * (mask > 1.e-5)
    i, j, k = np.nonzero(mask)
    x2, y2, z2 = np.max(i), np.max(j), np.max(k)
    x1, y1, z1 = np.min(i), np.min(j), np.min(k)
    dimx = x2 - x1
    dimy = y2 - y1
    dimz = z2 - z1
    if mmcif_file.endswith(".pdb"):
        pdb2mmcif(mmcif_file)
        mmcif_file = "./out.cif"
    doc = gemmi.cif.read_file(mmcif_file)
    block = doc.sole_block()  # cif file as a single block
    col_x = block.find_values("_atom_site.Cartn_x")
    col_y = block.find_values("_atom_site.Cartn_y")
    col_z = block.find_values("_atom_site.Cartn_z")
    if uc is not None:
        a, b, c = uc[2], uc[1], uc[0]
        alf = bet = gam = 90.
    else:
        a = block.find_value("_cell.length_a")
        b = block.find_value("_cell.length_b")
        c = block.find_value("_cell.length_c")
        alf = block.find_value("_cell.angle_alpha")
        bet = block.find_value("_cell.angle_beta")
        gam = block.find_value("_cell.angle_gamma")
    pixa = float(a) / mask.shape[0]
    pixb = float(b) / mask.shape[1]
    pixc = float(c) / mask.shape[2]
    cart_x = pixa * x1
    cart_y = pixb * y1
    cart_z = pixc * z1
    dim = np.max([dimz, dimy, dimx])
    if dim % 2 != 0:
        dim += 1
    dx = (dim - dimx) // 2
    dz = (dim - dimz) // 2
    dy = (dim - dimy) // 2
    for n, _ in enumerate(col_x):
        col_x[n] = str((float(col_x[n]) - cart_x) + (dx + padwidth) * pixa)
        col_y[n] = str((float(col_y[n]) - cart_y) + (dy + padwidth) * pixb)
        col_z[n] = str((float(col_z[n]) - cart_z) + (dz + padwidth) * pixc)
    doc.write_file("./tmp.cif")
    st = gemmi.read_structure("./tmp.cif")
    ca = (dim+padwidth*2) * pixa
    cb = (dim+padwidth*2) * pixb
    cc = (dim+padwidth*2) * pixc
    st.cell.set(ca, cb, cc, 90., 90., 90.)
    st.make_mmcif_document().write_file("emda_reboxed_model.cif")
    if os.path.isfile("out.cif"):
        os.remove("./out.cif")
    if os.path.isfile("tmp.cif"):
        os.remove("./tmp.cif")