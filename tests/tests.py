# this file contains emda2 tests
from os import name
import numpy as np
import emda2.emda_methods2 as em
from emda2.core import iotools
from emda.core import plotter

# read-write test
""" imap = "/Users/ranganaw/MRC/REFMAC/beta_galactosidase/fit/average/avgmap_0_unsharpened.mrc"
#imap = "/Users/ranganaw/MRC/REFMAC/COVID19/gesamt_test/emd_22255.map"
m1 = iotools.Map(name=imap)
m1.read()
m2 = iotools.Map(name='newmap.mrc')
m2.arr = m1.arr
m2.cell = m1.cell
m2.origin = m1.origin
m2.axorder = (2,1,0)#m1.axorder
m2.write()
exit() """

#1. power spectrum
""" imap = "/Users/ranganaw/MRC/REFMAC/ligand_challenge/EMD-7770/Trimmed_maps_and_model/emd_7770_trimmed.mrc"
#imap = "/Users/ranganaw/MRC/REFMAC/Check_mask/EMD-10563/emd_10563.map"
mapobj = iotools.Map(name=imap)
mapobj.read()
print('uc:', mapobj.cell)
print('nx, ny, nz: ', mapobj.arr.shape)
print('newcell: ', mapobj.workcell)
print('nx, ny, nz: ', mapobj.workarr.shape)
nbin, res_arr, bin_idx, sgrid = em.get_binidx(mapobj.workcell, mapobj.workarr)
power_spectrum = em.get_map_power(fo=np.fft.fftshift(np.fft.fftn(mapobj.workarr)),
            bin_idx=bin_idx, nbin=nbin)
plotter.plot_nlines_log(res_arr, [power_spectrum], 'power')
print("Resolution   bin     Power")
for i in range(len(res_arr)):
    print("{:.2f} {:.4f}".format(res_arr[i], power_spectrum[i]))   """

#2. FSC
""" imap1 = "/Users/ranganaw/MRC/REFMAC/ligand_challenge/EMD-7770/Trimmed_maps_and_model/emd_7770_half_map_1_trimmed.mrc"
imap2 = "/Users/ranganaw/MRC/REFMAC/ligand_challenge/EMD-7770/Trimmed_maps_and_model/emd_7770_half_map_2_trimmed.mrc"
#imap1 = "/Users/ranganaw/MRC/REFMAC/ligand_challenge/EMD-7770/EMD-7770/emd_7770_half1.map"
#imap2 = "/Users/ranganaw/MRC/REFMAC/ligand_challenge/EMD-7770/EMD-7770/emd_7770_half2.map"
mapobj1 = iotools.Map(name=imap1)
mapobj1.read()
mapobj2 = iotools.Map(name=imap2)
mapobj2.read()
nbin, res_arr, bin_idx = em.get_binidx(mapobj1.cell, mapobj1.arr)
twomapfsc = em.fsc(f1=np.fft.fftshift(np.fft.fftn(mapobj1.arr)),
                f2=np.fft.fftshift(np.fft.fftn(mapobj2.arr)),
                bin_idx=bin_idx,
                nbin=nbin)
plotter.plot_nlines(res_arr=res_arr,
                    list_arr=[twomapfsc],
                    mapname="twomap_fsc.eps",) """ 
#2.1 nFSC
""" path = "/Users/ranganaw/MRC/REFMAC/Takanori_ATPase/fitting_structures/emdafit/"
staticmap = "static_map_9931.mrc"
maplist = [
        "transformed_fullmap_9932.mrc",
        "emda_fitted_9933.mrc",
        "transformed_fullmap_9934.mrc",
        "transformed_fullmap_9935.mrc",
        "emda_fitted_9936.mrc",
        "transformed_fullmap_9937.mrc",
        "emda_magcorretedmap_9938.mrc",
        "emda_magcorretedmap_9939.mrc",
        "emda_magcorretedmap_9940.mrc",
        "transformed_fullmap_9941.mrc",
        "transformed_fullmap_9942.mrc",
]
fsclist = []
labels = ["9932", "9933", "9934", "9935", "9936", "9937", "9938", "9939", "9940", "9941", "9942"]
stmap = iotools.Map(name=path+staticmap)
stmap.read()
nbin, res_arr, bin_idx, _ = em.get_binidx(stmap.cell, stmap.arr)
for imap in maplist:
    mobj = iotools.Map(name=path+imap)
    mobj.read()
    twomapfsc = em.fsc(f1=np.fft.fftshift(np.fft.fftn(stmap.arr)),
                    f2=np.fft.fftshift(np.fft.fftn(mobj.arr)),
                    bin_idx=bin_idx,
                    nbin=nbin)
    fsclist.append(twomapfsc)
plotter.plot_nlines(res_arr=res_arr,
                    list_arr=fsclist,
                    mapname="all_fsc.eps",
                    curve_label=labels,
                    fscline=0.,
                    plot_title="FSC against EMD-9931 using fullmaps") """
    



# 3. RCC
""" path = "/Users/ranganaw/MRC/REFMAC/ligand_challenge/EMD-7770/Trimmed_maps_and_model/"
half1 = path + "emd_7770_half_map_1_trimmed.mrc"
half2 = path + "emd_7770_half_map_2_trimmed.mrc"
modelname = path + "6cvm_trimmed.cif"
mapobj1 = iotools.Map(name=half1)
mapobj1.read()
mapobj2 = iotools.Map(name=half2)
mapobj2.read()
# generate modelbased map
model = em.model2map_gm(modelxyz=modelname, 
                        resol=4, 
                        dim=mapobj1.arr.shape, 
                        cell=mapobj1.cell, 
                        maporigin=mapobj1.origin)
#print(model.shape)
mapobj = iotools.Map(name="modelmap.mrc")
mapobj.arr = model #np.transpose(model)
mapobj.cell = mapobj1.cell
mapobj.origin = mapobj1.origin
mapobj.axorder = mapobj1.axorder
mapobj.write()
# RCC calculation
rcc = em.realsp_correlation(arr_hf1=mapobj1.arr,
                            arr_hf2=mapobj2.arr,
                            uc=mapobj1.cell,
                            norm=True,
                            model=model, #np.transpose(model),
                            origin=mapobj1.origin,
                            axorder=mapobj1.axorder) """

# 4. resampling an array
""" imap = "/Users/ranganaw/MRC/REFMAC/ligand_challenge/EMD-7770/EMD-7770/emd_7770.map"
mapobj1 = iotools.Map(name=imap)
mapobj1.read()
target_pix = [1., 1., 1.]
resampled_arr  = em.resample_data(curnt_pix=[mapobj1.cell[i]/shape for i, shape in enumerate(mapobj1.arr.shape)],
                                 targt_pix=target_pix,
                                 arr=mapobj1.arr,
                                 #targt_dim=[216,216,216],
                                  )
mapobj = iotools.Map("resampled.mrc")
mapobj.arr = resampled_arr
mapobj.cell = [target_pix[i]*shape for i, shape in enumerate(resampled_arr.shape)]
mapobj.origin = mapobj1.origin
mapobj.axorder = mapobj1.axorder
mapobj.write() """

# 5. Map 2 MTZ
""" imap = "emd_7770_trimmed.mrc"
mapobj1 = iotools.Map(name=imap)
mapobj1.read()
em.map2mtz(arr=mapobj1.arr, uc=mapobj1.cell, mtzname="map2mtz_noncubic.mtz", resol=0.) """

# 6. Mtz to Map
""" imap = "./emd_7770_trimmed.mrc"
#imap = "./emd_7770.map"
mapobj1 = iotools.Map(name=imap)
mapobj1.read()
mtzname = "./map2mtz_noncubic.mtz"
#mtzname = "./map2mtz_cubic.mtz"
print(mtzname)
arr, uc = em.mtz2map(mtzname=mtzname, map_size=mapobj1.arr.shape)
mapobj = iotools.Map(name="mtz2map_noncubic.map")
mapobj.arr = arr
mapobj.cell = uc #mapobj1.cell
mapobj.origin = mapobj1.origin
mapobj.axorder = mapobj1.axorder
mapobj.write() """

# 7. Real space map vs model correlation
""" path = "/Users/ranganaw/MRC/REFMAC/ligand_challenge/EMD-7770/Trimmed_maps_and_model/"
imap = path + "emd_7770_half_map_1_trimmed.mrc"
#half2 = path + "emd_7770_half_map_2_trimmed.mrc"
modelname = path + "6cvm_trimmed.cif"
mapobj1 = iotools.Map(name=imap)
mapobj1.read()
model = em.model2map_gm(modelxyz=modelname, 
                        resol=4, 
                        dim=mapobj1.arr.shape, 
                        cell=mapobj1.cell, 
                        maporigin=mapobj1.origin)
# mask from coordinates
maskobj = em.mask_from_atomic_model(mapname=imap, 
                                   modelname=modelname, 
                                   atmrad=5)
#mask = maskobj.arr
#maskobj.name = "emda_atomic_mask.mrc"
#maskobj.write()
#print(mask.shape)
mapmodel_rcc, kern_rad = em.realsp_correlation_mapmodel(
                            uc=mapobj1.cell,
                            map=mapobj1.arr,
                            model=model,
                            resol=4,
                            mask=maskobj.arr,
                            )
mapobj = iotools.Map(name="rcc_modelmap_kernx.mrc")
mapobj.arr = mapmodel_rcc
mapobj.cell = mapobj1.cell
mapobj.origin = mapobj1.origin
mapobj.axorder = mapobj1.axorder
mapobj.write() """

# 8. Overlay
""" #path = "/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3651/other/"
maplist = [
    #"./emd_3651.map",
    "static_map.mrc",
    #"./emd_3651_rotatedmap_10deg.mrc",
    #"./emd_3651_tra1A.mrc",
    #"emd_3651_rotatedmap_10deg_tra3A.mrc",
    "fitted_map_1.mrc"
]
arrlist = []
pixlist = []
cells = []
origins = []
for imap in maplist:
    m = iotools.Map(name=imap)
    m.read()
    arrlist.append(m.arr)
    cells.append(m.cell)
    origins.append(m.origin)
    pixlist.append([m.cell[j] / dim for j, dim in enumerate(m.arr.shape)])

_,_,_ = em.overlay(arrlist=arrlist, 
                   pixlist=pixlist, 
                   cell=cells[0], 
                   origin=origins[0], 
                   ) """

#9. Lowpass filter
""" imap = "/Users/ranganaw/MRC/REFMAC/Check_mask/EMD-10563/emd_10563.map"
m1 = iotools.Map(name=imap)
m1.read()
_, lwp = em.lowpass_map(uc=m1.workcell, arr1=m1.workarr, resol=5, filter="butterworth")
m2 = iotools.Map(name="lwp.mrc")
croppedImage = iotools.cropimage(arr=lwp, tdim=m1.arr.shape)
m2.cell = m1.cell
m2.arr = croppedImage
m2.axorder = m1.axorder
m2.write()
exit() """

# 10. Mask from Map
""" imap = "/Users/ranganaw/MRC/REFMAC/Check_mask/EMD-10563/emd_10563.map"
m1 = iotools.Map(name=imap)
m1.read()
mask, lwp = em.mask_from_map(uc=m1.workcell, arr=m1.workarr, resol=10., filter='butterworth')
croppedImage = iotools.cropimage(arr=mask, tdim=m1.arr.shape)
m2 = iotools.Map(name='mapmask.mrc')
m2.cell = m1.cell
m2.arr = croppedImage
m2.origin = m1.origin
m2.axorder = m1.axorder
m2.write() """

# 11. Mask from Map using connected pixels

#12. rotation center of the map
imap = "/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3651/other/rotation_centre/emda_rbxfullmap_emd-3651.mrc"
imask = "/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3651/other/rotation_centre/emda_rbxmapmask_emd-3651.mrc"
axis = [2.94523737e-03, 2.89148106e-06, 9.99995663e-01]
order = 2
resol = 4.
m1 = iotools.Map(imap)
m1.read()
mm = iotools.Map(imask)
mm.read()
rc = em.get_rotation_center(m1=m1, mm=mm, axis=axis, order=order, resol=resol)
print(rc)


