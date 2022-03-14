import gemmi


def pdb2mmcif(filename_pdb):
    structure = gemmi.read_structure(filename_pdb)
    structure.setup_entities()
    structure.assign_label_seq_id()
    mmcif = structure.make_mmcif_document()
    mmcif.write_file("out.cif")