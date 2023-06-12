import emda2.emda_methods2 as em

imap = "./emd_6952_fullmap.mrc"  # "./emd_6952.map"
imask = "./emd_6952_msk_1.map"
resol = 4.25

imap = "/Users/Rangana/EMDA2_test/EMD-12680/emd_12680.map"
imask = "/Users/Rangana/EMDA2_test/EMD-12680/mapmask.mrc"
resol = 2.5


_ = em.get_pointgroup(
    half1="",  # "./emd_6952_half_map_1.map",
    half2="",  # "./emd_6952_half_map_2.map",
    pmap=imap,
    mask=imask,
    resol=resol,
    output_maps=False,
)
