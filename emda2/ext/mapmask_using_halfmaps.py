import numpy as np
import re
import emda2.emda_methods2 as em
import fcodes2
from emda2.core import iotools, plotter, maptools
from numpy.fft import fftn, fftshift, ifftshift, ifftn
from emda2.ext.utils import filter_fsc #, get_ibin 
#from loc_averaging_anyradius import f_sphere
from emda2.ext.mapmask import binary_closing, binary_dilation, globular_mask
from emda2.ext.maskmap_class import make_soft
from more_itertools import sort_together
from skimage import measure
from emda2.ext.sym.symanalysis_pipeline import writemap

#
from emda2.core import restools
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

from emda.ext.mapfit import rdp_algo


def create_mapmask_islandlabelling2(rho1, rho1avg, thresh):
    if np.isscalar(thresh):
        arr = rho1avg >= thresh
    else:
        arr = thresh
    blobs = arr
    radius = max(blobs.shape) // 2
    gmask = globular_mask(arr=blobs, radius=radius, com=True)
    blobs = blobs * gmask
    blobs_labels, nlabels = measure.label(blobs, background=0, connectivity=blobs.ndim, return_num=True)
    print('blob assignment done')
    print('# labels: ', nlabels)
    regionprops = measure.regionprops(blobs_labels)
    # new code 13June2022
    blob_number = []
    blob_area = []
    bsum = 0
    for i in range(nlabels):
        blob_number.append(i+1)
        blob_area.append(regionprops[i].area)
        bsum += regionprops[i].area
    sblob_area, sblob_number = sort_together([blob_area, blob_number], reverse=True)
    bnum_highvol = []
    bsum_highvol = []
    rmsd_list = []
    for i in range(nlabels):
        vol_frac = sblob_area[i]/bsum
        if vol_frac >= 0.05:
            bnum_highvol.append(sblob_number[i])
            xx = rho1 * (blobs_labels == sblob_number[i])
            bsum_highvol.append(np.sum(xx))
            rmsd = np.sqrt(np.mean((xx - np.mean(xx))**2))
            rmsd_list.append(rmsd)
            print(sblob_number[i], np.sum(xx), vol_frac, rmsd)

    print('sorting...')
    srmsd_list, sbnum_highvol = sort_together([rmsd_list, bnum_highvol], reverse=True)

    mask_list = []
    for num in sbnum_highvol:
        print('making masks...')
        mask = blobs * (blobs_labels == num)
        mask_list.append(mask) # for testing undilated mask
        #nmask = binary_closing(mask * gmask)
        #nmask = binary_dilation(nmask, 9) 
        #nmask = make_soft(nmask, 3)
        #mask_list.append(nmask)    
    return mask_list


def volume_of_sphere(arr):
    nx, ny, nz = arr.shape
    r = nx // 2
    vol = 4 / 3 * np.pi * r**3
    return vol


def find_vf(arr, plotname='plot.png'):
    nx, ny, nz = arr.shape
    vol = nx * ny * nz
    vflist = []
    threshlist = [0.005 * i for i in range(1, 201)]
    arr = arr / np.amax(arr)
    tvf = 0.
    tvflist  = []
    for i in range(1, 201):
        binarymask = arr >= (i * 0.005)
        vol_bmask = np.sum(binarymask)
        vf = vol_bmask / vol
        tvf += vf
        tvflist.append(tvf)
        vflist.append(vf)
        print('i, vf: ', i, vf)
    #print(np.diff(np.asarray(vflist)))
    #plt.plot(np.diff(np.asarray(vflist)))
    #plt.show()
    ibin = np.argmax(np.abs(np.diff(np.asarray(vflist)))) #+ 1
    print('vf = ', vflist[ibin])
    print('vfb: ', 0.005*ibin)
    xc, yc = rdp_algo.run_rdp(np.asarray(threshlist), np.asarray(vflist), epsilon=0.005)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(threshlist, vflist)
    #ax1.plot(threshlist, tvflist)
    ax1.plot(xc, yc, marker='o')
    ax1.set_xlabel('threshold')
    ax1.set_ylabel('volume fraction')
    ax1.set_title("Volume fraction against the threshold of the variance map")
    plt.savefig(plotname, format="png", dpi=300)
    #plt.show()
    #binarymask = arr >= (1. - 0.005*ibin)
    print('xc=', xc)
    if len(xc) == 2:
        thresh = (xc[0] + xc[1]) / 2
    else:
        thresh = xc[1]
    binarymask = arr >= thresh
    return binarymask, thresh


def variance40_mask(arr, vfthresh=0.4):
    nx, ny, nz = arr.shape
    vol = nx * ny * nz
    arr = arr / np.amax(arr)
    for i in range(1, 201):
        binarymask = arr >= (1. - i * 0.005)
        vol_bmask = np.sum(binarymask)
        vf = vol_bmask / vol
        print('i, vf: ', i, vf)
        if vf > vfthresh:
            break
    return binarymask


def thresholded_by_volume(arr, thresh=0.005):
    nx, ny, nz = arr.shape
    vol = nx * ny * nz
    #vol_sp = volume_of_sphere(arr)
    arr = arr / np.amax(arr)
    binarymask = arr >= (1. - thresh)
    vol_bmask = np.sum(binarymask)
    #print('vol fraction from box: ', vol_bmask / vol)
    #print('vol fraction from sphere: ', vol_bmask / vol_sp)
    vflist = []
    while 1:
        binarymask = arr >= (1. - thresh)
        vol_bmask = np.sum(binarymask)
        vf = vol_bmask / vol
        vflist.append(vf)
        print('vf: ', vf)
        if vf > 0.03:
            print('volfrac: ', thresh)
            break
        else:
            thresh += 0.005
    plt.plot(vflist)
    plt.show()
    return binarymask

def cdf_thresholding(arr, prob=0.99, plotname='cdf.png', vthresh=0.):
    import matplotlib
    matplotlib.use(matplotlib.get_backend())
    import matplotlib.pyplot as plt
    arr = arr / np.amax(arr)
    X2 = np.sort(arr.flatten())
    F2 = np.array(range(len(X2))) / float(len(X2) - 1)
    xc, yc = rdp_algo.run_rdp(X2, F2, epsilon=0.01)
    print(xc)
    print(yc)
    loc = np.where(F2 >= prob)
    thresh = X2[loc[0][0]]
    print('threshold: ', thresh)
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.plot(X2, F2)
    ax2.plot(xc, yc, marker='o')
    m1, c1, ynew = fit_a_line(X2, F2)
    ax2.plot(X2, ynew)
    m2, c2, xc, ynew = fit_a_line2(X2, F2)
    ax2.plot(xc, ynew)
    xi, yi = find_intersection_point(m1, m2, c1, c2)
    print('intersection point (x, y): ', xi, yi)
    ax2.plot(xi, yi, marker="x")
    ax2.plot((vthresh, vthresh), (0.0, 1.0), color="black", linestyle=":")
    vallist = []
    for i, _ in enumerate(X2):
        val = (X2[i] - vthresh)**2
        vallist.append(val)
    ibin = np.argmin(np.asarray(vallist))
    print('ibin: ', ibin)
    print('Corresponding CDF value at ibin: ', F2[ibin])
    ax2.plot((X2[0], X2[-1]), (F2[ibin], F2[ibin]), color="black", linestyle=":")
    yval = "{:.3f}".format(F2[ibin])
    ax2.text(0.0, F2[ibin], yval, size=10)
    ax2.set_xlabel("$X$")
    ax2.set_ylabel("$CDF$")
    ax2.set_title("CDF against threshold of the variance map")
    plt.savefig(plotname, format="png", dpi=300)
    #plt.show()
    thresh = vthresh
    return arr >= thresh


def find_intersection_point(m1, m2, c1, c2):
    xi = (c1 - c2) / (m2 - m1)
    yi = m1 * xi + c1
    return xi, yi


def fit_a_line(x, y):
    nx = np.size(x, 0)
    npoints = nx - int(nx*0.99)
    print('# points for fitting: ', npoints)
    xc = x[int(nx*0.99) : nx]
    yc = y[int(nx*0.99) : nx]
    print(xc)
    print(yc)
    print('fitting bestline ...')
    a, b = np.polyfit(xc, yc, 1)
    ynew =  a*x + b
    return a, b, ynew

def fit_a_line2(x, y):
    ny = np.size(y, 0)
    print('npoints: ', ny)
    npoints = int(ny*0.5)
    print('# points for fitting: ', npoints)
    yc = y[0 : npoints]
    xc = x[0 : npoints]
    print(xc)
    print(yc)
    print('fitting bestline ...')
    a, b = np.polyfit(xc, yc, 1)
    print(a, b)
    ynew =  a*x + b
    return a, b, x[ynew<1], ynew[ynew<1]

def fit_a_line3(x, y):
    ny = np.size(y, 0)
    print('npoints: ', ny)
    npoints = int(ny*0.9)
    print('# points for fitting: ', npoints)
    yc = y[0 : npoints]
    xc = x[0 : npoints]
    print(xc)
    print(yc)
    print('fitting bestline ...')
    a, b = np.polyfit(yc, xc, 1)
    print(a, b)
    xnew = a*y + b
    return a, b, xnew

def get_ibin(bin_fsc, cutoff):
    cutoff1 = cutoff
    #rounded_fsc = np.around(bin_fsc, 2)
    #from scipy import stats
    #mode, count = stats.mode(rounded_fsc[:5])
    #cutoff2 = mode[0] * 0.8
    cutoff2 = np.amax(bin_fsc) * 0.8
    cutoff = min([cutoff1, cutoff2])
    print('cutoff: ', cutoff)
    # search from rear end
    ibin = 0
    for i, ifsc in reversed(list(enumerate(bin_fsc))):
        if ifsc > cutoff:
            ibin = i
            if ibin % 2 != 0:
                ibin = ibin - 1
            break
    #print('get_ibin, ibin: ', ibin)
    return ibin

def main(h1, h2, emdbid):
    """ m = re.search('emd_(.+)_half', half1)
    emdbid = 'emd_%s'%m.group(1)
    plotname_var = 'cdf_var_emd_%s.png'%m.group(1)
    plotname_rho = 'cdf_rho_emd_%s.png'%m.group(1)
    maskname = 'emdamapmask_emd-%s.mrc'%m.group(1)
    half2 = half1.replace("half_map_1", "half_map_2")
    h1 = iotools.Map(half1)
    h1.read()
    h2 = iotools.Map(half2)
    h2.read() """    

    f1 = fftshift(fftn(fftshift(h1.workarr)))
    f2 = fftshift(fftn(fftshift(h2.workarr)))

    pixsize = h1.workcell[0] / h1.workarr.shape[0]

    # calculate halfmap FSC
    nbin, res_arr, bin_idx, sgrid = em.get_binidx(
        cell=h1.workcell, arr=h1.workarr)
    halffsc = em.fsc(
        f1=f1, f2=f2, bin_idx=bin_idx, nbin=nbin)
    ibin = get_ibin(filter_fsc(halffsc), cutoff=0.75)
    smax = res_arr[ibin]
    # compute NEM
    fsc_str = np.sqrt(filter_fsc(2*halffsc / (1.0 + halffsc), thresh=0.75))
    fsc_str_halfmap = np.sqrt(filter_fsc(halffsc, thresh=0.75))
    plotter.plot_nlines(
        res_arr=res_arr,
        list_arr=[fsc_str],
        )
    nx, ny, nz = f1.shape
    e1 = fcodes2.get_normalized_sf_singlemap(
        fo=f1,
        bin_idx=bin_idx,
        nbin=nbin,
        mode=0,
        nx=nx,ny=ny,nz=nz,
        )
    e2 = fcodes2.get_normalized_sf_singlemap(
        fo=f2,
        bin_idx=bin_idx,
        nbin=nbin,
        mode=0,
        nx=nx,ny=ny,nz=nz,
        )
    #fscweights = fcodes2.read_into_grid(bin_idx, fsc_str, nbin, nx, ny, nz)
    fscweights_halfmap = fcodes2.read_into_grid(bin_idx, fsc_str_halfmap, nbin, nx, ny, nz)
    #writemap(fscweights, h1.workcell, emdbid+'_weights.mrc', h1.origin)
    e1 = e1 * fscweights_halfmap 
    e2 = e2 * fscweights_halfmap
    # ploting power spectrum
    """ ps1 = maptools.get_map_power(fo=e1, bin_idx=bin_idx, nbin=nbin)
    ps2 = maptools.get_map_power(fo=e2, bin_idx=bin_idx, nbin=nbin)
    for i in range(nbin):
        print(res_arr[i], ps1[i], ps2[i]) """
    rho1 = np.real(ifftshift(ifftn(ifftshift(e1))))
    rho2 = np.real(ifftshift(ifftn(ifftshift(e2))))
    rho = (rho1+rho2)/2
    #writemap(rho, h1.workcell, emdbid+'_rho.mrc', h1.origin)
    #noise = (rho1 - rho2)/2
    kernsize = int((smax * 3) / pixsize)
    print('kernsize:', kernsize)
    kern = restools.create_soft_edged_kernel_pxl(kernsize)
    # replace small and negative values by average
    rho = np.where(rho < np.average(rho), np.average(rho), rho)
    loc3_A = fftconvolve(rho, kern, "same")
    loc3_A2 = fftconvolve(rho * rho, kern, "same")
    var3_A = loc3_A2 - loc3_A ** 2
    #writemap(var3_A, h1.workcell, emdbid+'_rhovar_unmasked.mrc')
    rhovar = var3_A  
    volmask, vthresh = find_vf(arr=rhovar, plotname=emdbid+"_vf.png")
    # island detection
    masklist = create_mapmask_islandlabelling2(
        rho1=rho, rho1avg=volmask, thresh=volmask)
    volmask = masklist[0] # just get the first mask in the list
    #writemap(volmask, h1.workcell, emdbid+'_volmask_binary.mrc')
    # need to dilate and soft
    #print('Binar operations are carrying out ...')
    mask = binary_dilation(binary_closing(volmask), 9) 
    mask = make_soft(mask, 3)
    #print('Outputting %s_volmask.mrc' %emdbid)
    #writemap(mask, h1.workcell, emdbid+'_volmask.mrc')
    print('Done!')

    # re-calculating stats using masked maps
    """ rho1 = rho1 * volmask
    rho2 = rho2 * volmask
    rho = (rho1 + rho2) / 2
    noise = (rho1 - rho2) / 2
    # on rho
    rhoavg = fftconvolve(rho, kern, "same")
    #writemap(rhoavg, h1.workcell, emdbid+'_rhoavg_emda.mrc')
    rho2avg = fftconvolve(rho * rho, kern, "same")
    rhovar = rho2avg - rhoavg ** 2
    writemap(var3_A, h1.workcell, emdbid+'_rhovar_masked.mrc') 
    # on noise
    noise_avg = fftconvolve(noise, kern, "same")
    #writemap(noise_avg, h1.workcell, emdbid+'_noiseavg.mrc')
    noise2_avg = fftconvolve(noise * noise, kern, "same")
    noise_var = noise2_avg - noise_avg ** 2
    writemap(noise_var, h1.workcell, emdbid+'_noisevar_masked.mrc')
    # local cc
    tmp = (rhovar - noise_var)
    regval = np.amax(rhovar) / 1000
    rhovar = np.where(rhovar < regval, regval, rhovar)
    #writemap(rhovar_ma, h1.workcell, emdbid+'_rhovar_masked.mrc')
    localcc = (tmp/rhovar)
    writemap(localcc, h1.workcell, emdbid+'_localcc_masked.mrc') """
    return mask





if __name__=="__main__":
    halfmaplist = [
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-6952/emd_6952_half_map_1.map", #C3
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10561/emd_10561_half_map_1.map", #C35
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0153/emd_0153_half_map_1.map", #D2
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-25437/emd_25437_half_map_1.map", #T
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0263/emd_0263_half_map_1.map", #O
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12181/emd_12181_half_map_1.map", #I
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-12139/reboxed_emd_12139_half_map_1.mrc",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-23884/emd_23884_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21185/emd_21185_half_map_1.map", # too bog for Mac Mem
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10461/emd_10461_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-20690/emd_20690_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-21246/emd_21246_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-23610/emd_23610_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-23884/emd_23884_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21231/emd_21231_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-4906/emd_4906_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-11220/emd_11220_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12073/emd_12073_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-12608/emd_12608_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12819/emd_12819_half_map_1.map",
        "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12822/emd_12822_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12707/emd_12707_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12644/emd_12644_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-13624/emd_13624_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-13844/emd_13844_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-7446/emd_7446_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-9953/emd_9953_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0121/emd_0121_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10325/emd_10325_half_map_1.map"
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12605/emd_12605_half_map_1.map"

    ]

    for ihalf in halfmaplist:
        main(ihalf)

