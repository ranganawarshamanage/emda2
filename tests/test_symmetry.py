import emda2.emda_methods2 as em

half1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21231/emd_21231_half_map_1.map"
mask = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21231/emd_21231_msk_1.map"
resol = 6.0

try:
    (proshade_pg, 
    emda_pg, 
    maskname, 
    reboxedmapname) = em.get_pointgroup(
        half1=half1,
        mask=mask,
        resol=float(resol)
    )
except ValueError as e:
    print(e)
