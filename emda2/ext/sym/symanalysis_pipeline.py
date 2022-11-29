
"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
# Symmetry ananlysis and pointgroup detection
# Halfmaps are required
# Compute FSC-hf --> FSC-full
# Compute FSC-sym from fullmap and sym.copy
# Compare FSC-full vs FSC-sym
import numpy as np
import re, math, os
from numpy.fft import fftn, ifftn, fftshift
from emda2.core import iotools
import emda2.emda_methods2 as em
from emda2.ext.mapmask import mask_from_halfmaps
from emda2.ext.sym import proshade_tools
from emda2.ext.utils import (
    get_ibin, 
    filter_fsc,
    shift_density, 
    center_of_mass_density,
    lowpassmap_butterworth)
import fcodes2
from emda2.ext.sym import axis_refinement
from more_itertools import sort_together
from emda2.ext.sym import pgcode
from emda2.ext.sym.download_halfmaps import fetch_halfmaps
from emda2.core import emdalogger
from emda2.core.emdalogger import vec2string, list2string
from emda2.ext.sym.decide_pointgroup import decide_pointgroup
import emda2.ext.sym.average_symcopies as avgsym

def writemap(arr, cell, mapname, origin=None):
    # write map into a file
    if origin is None:
        origin = [0, 0, 0]
    m2 = iotools.Map(name=mapname)
    m2.arr = arr
    m2.cell = cell
    m2.origin = origin
    m2.write()

def is_prime(n):
    """
    https://stackoverflow.com/questions/15285534/isprime-function-for-python-language
    """
    if n == 2 or n == 3: return True
    if n < 2 or n%2 == 0: return False
    if n < 9: return True
    if n%3 == 0: return False
    r = int(n**0.5)
    f = 5
    while f <= r:
        if n % f == 0: return False
        if n % (f+2) == 0: return False
        f += 6
    return True 

def normalise_axis(axis):
    ax = np.asarray(axis, 'float')
    return ax / math.sqrt(np.dot(ax, ax))

def test(emmap1, axes, orders, fscs, fobj=None):
    if fobj is None:
        fobj = open('pointgroup.txt', 'w')
    ang_tol_p = 3.0 # angle tolerence for proshade axes
    fsc_thresh_for_axrefine = 0.95 # FCS threshold for axis refinement 
    pg_decide_fsc = 0.9 # FSC threshold for pointgroup decision
    emdalogger.log_string(
        fobj, "\n *** Default parameters used ***")
    emdalogger.log_string(
        fobj, "Angle tolerence for Proshade axes:     %s (deg)" %ang_tol_p)
    emdalogger.log_string(
        fobj, "FCS threshold for axis refinement:     %s" %fsc_thresh_for_axrefine)
    emdalogger.log_string(
        fobj, "FSC threshold for pointgroup decision: %s" %pg_decide_fsc)
    # choose only prime order axes with +ve FSCs
    prime_orderlist = []
    prime_axlist = []
    prime_fsclist = []
    fsc_tmp_list = []
    for i, order in enumerate(orders):
        if is_prime(order):
            fsc_tmp_list.append(fscs[i])
    fsc_max = max(fsc_tmp_list)
    emdalogger.log_string(fobj, '\n    *** Chosen prime axes ***')
    emdalogger.log_string(fobj, '--------------------------------------')
    emdalogger.log_string(fobj, ' #      X      Y      Z   Order   FSC')
    emdalogger.log_string(fobj, '--------------------------------------')
    min_fsc_for_primeaxes_detect = 0.2
    for i, order in enumerate(orders):
        if is_prime(order):
            #if fscs[i] > max(min_fsc_for_primeaxes_detect, fsc_max*0.2):
            if fscs[i] > 0.0:
                prime_orderlist.append(order)
                ax = normalise_axis(axes[i])
                prime_axlist.append(ax)
                prime_fsclist.append(fscs[i])
                emdalogger.log_string(
                    fobj,
                    "%i   %s   %s  % .3f"%(
                    i, vec2string(ax), int(order), fscs[i])
                )
    emdalogger.log_string(fobj, '--------------------------------------')
 
    if len(prime_orderlist) == 0:
        pg = 'C1'
        return pg
    order5ax, order5fsc = [], []
    order3ax, order3fsc = [], []
    order2ax, order2fsc = [], []
    ordernax, ordernfsc, ordern = [], [], []
    for i, order in enumerate(prime_orderlist):
        if order == 5:
            order5ax.append(prime_axlist[i])
            order5fsc.append(prime_fsclist[i])
        elif order == 3:
            order3ax.append(prime_axlist[i])
            order3fsc.append(prime_fsclist[i])
        elif order == 2:
            order2ax.append(prime_axlist[i])
            order2fsc.append(prime_fsclist[i])
        else:
            ordernax.append(prime_axlist[i])
            ordernfsc.append(prime_fsclist[i])
            ordern.append(prime_orderlist[i])
    print("prime axes obtained")
    # sort by FSC
    if len(order5ax) > 1:
        sorder5fsc, sorder5ax = sort_together([order5fsc, order5ax], reverse=True)
    else:
        sorder5fsc, sorder5ax = order5fsc, order5ax
    if len(order3ax) > 1:
        sorder3fsc, sorder3ax = sort_together([order3fsc, order3ax], reverse=True)
    else:
        sorder3fsc, sorder3ax = order3fsc, order3ax
    if len(order2ax) > 1:
        sorder2fsc, sorder2ax = sort_together([order2fsc, order2ax], reverse=True)
    else:
        sorder2fsc, sorder2ax = order2fsc, order2ax
    if len(ordernax) > 1:
        sordernfsc, sordernax, sordern = sort_together(
            [ordernfsc, ordernax, ordern], reverse=True)
    else:
        sordernfsc, sordernax, sordern = ordernfsc, ordernax, ordern

    emdalogger.log_newline(fobj)
    emdalogger.log_string(fobj, "Number of 5-fold axes: %s" %len(sorder5ax))
    emdalogger.log_string(fobj, "Number of 3-fold axes: %s" %len(sorder3ax))
    emdalogger.log_string(fobj, "Number of 2-fold axes: %s" %len(sorder2ax))
    emdalogger.log_string(fobj, "Number of n-fold axes: %s" %len(sordernax))

    emdalogger.log_string(fobj, "\n        Detecting Pointgroup        ")
    emdalogger.log_string(fobj, '--------------------------------------')

    pg = 'C1'
    if len(sorder5ax) > 0:
        axes = [sorder5ax, sorder3ax, sorder2ax, sordernax]
        fscs = [sorder5fsc, sorder3fsc, sorder2fsc, sordernfsc]
        pg = pgcode.five_folds(
            emmap1=emmap1, axes=axes, fscs=fscs, sordern=sordern, fobj=fobj)
    else:
        if len(sorder3ax) > 0:
            axes = [sorder3ax, sorder2ax, sordernax]
            fscs = [sorder3fsc, sorder2fsc, sordernfsc] 
            pg = pgcode.no5folds(
                emmap1=emmap1, axes=axes, fscs=fscs, sordern=sordern, fobj=fobj)
        else:
            if len(sordernax) > 0:
                axes = [sorder2ax, sordernax]
                fscs = [sorder2fsc, sordernfsc]     
                pg = pgcode.no53folds(
                    emmap1=emmap1, axes=axes, fscs=fscs, sordern=sordern, fobj=fobj)
            else:
                pg = pgcode.just2folds(
                    emmap1=emmap1, sorder2ax=sorder2ax, sorder2fsc=sorder2fsc, fobj=fobj)
    return pg


def get_pg(axlist, orderlist, fsclist, m1, claimed_res, **kwargs):
    res_arr = kwargs['res_arr']
    bin_idx = kwargs['bin_idx']
    nbin = kwargs['nbin']
    sgrid = kwargs['sgrid']
    fobj = kwargs['fobj']
    fsc_full = kwargs['fsc_full']
    mask = kwargs['mask']
    half1 = kwargs['half1']
    half2 = kwargs['half2']
    if claimed_res < 5:
        resol4refinement = float(5)
    else:
        resol4refinement = claimed_res
    # select data upto claimed resol
    _, map1 = lowpassmap_butterworth(
        fc=m1.workarr, 
        sgrid=sgrid,
        smax=claimed_res, 
        order=4)
    map1 = map1 * mask
    # using COM
    nx, ny, nz = map1.shape
    com = center_of_mass_density(map1)
    box_centr = (nx // 2, ny // 2, nz // 2)
    map1_com_adjusted = shift_density(map1, np.subtract(box_centr, com))
    mask_com_adjusted = shift_density(mask, np.subtract(box_centr, com))

    # preparing data for sym. averaging
    half1masked = half1 * mask
    half2masked = half2 * mask
    half1_com_adjusted = shift_density(half1masked, np.subtract(box_centr, com))
    half2_com_adjusted = shift_density(half2masked, np.subtract(box_centr, com))
    fhf1 = fftshift(fftn(fftshift(half1_com_adjusted)))
    fhf2 = fftshift(fftn(fftshift(half2_com_adjusted)))
    #
    com = center_of_mass_density(map1_com_adjusted)
    fo = fftshift(fftn(fftshift(map1_com_adjusted)))
    # make NEM
    eo = fcodes2.get_normalized_sf_singlemap(
        fo=fo,
        bin_idx=bin_idx,
        nbin=nbin,
        mode=0,
        nx=nx,ny=ny,nz=nz,
        )
    fsc_max = np.sqrt(filter_fsc(fsc_full))
    eo = eo * fcodes2.read_into_grid(bin_idx, fsc_max, nbin, nx, ny, nz)
    #eo = fcodes2.simple_blur(eo, m1.workcell, nx, ny, nz) #simple blurring
    dist = np.sqrt((res_arr - claimed_res) ** 2)
    claimed_cbin = np.argmin(dist) + 1  
    tlist = [[0., 0., 0.] for _ in range(len(axlist))]
    emmap1 = axis_refinement.EmmapOverlay(arr=map1_com_adjusted)
    emmap1.bin_idx = bin_idx
    emmap1.res_arr = res_arr
    emmap1.nbin = nbin
    emmap1.claimed_res = claimed_res
    emmap1.claimed_bin = claimed_cbin
    emmap1.mask = mask_com_adjusted
    emmap1.pix = [m1.workcell[i]/sh for i, sh in enumerate(m1.workarr.shape)]
    emmap1.fo_lst = [fo]
    emmap1.eo_lst = [eo]
    emmap1.fitres = resol4refinement
    emmap1.fitfsc = 0.15
    emmap1.ncycles = 10
    emmap1.fscfull = fsc_full
    emmap1.symdat = []
    emmap1.syminfo = []
    emmap1.com = True
    emmap1.com1 = com
    emmap1.map_dim = eo.shape
    emmap1.map_unit_cell = m1.workcell
    emdalogger.log_vector(
        fobj, 
        {'Centre of Mass [x, y, z] (pixel units)':list(com)})
    emmap1.pix = [m1.workcell[i]/sh for i, sh in enumerate(m1.workarr.shape)]
    emdalogger.log_vector(
        fobj, 
        {'Centre of Mass [x, y, z] (A)':[com[i]*emmap1.pix[i] for i in range(3)]})
    print('Finding pointgroup...')
    pg = test(
        emmap1=emmap1, 
        axes=axlist, 
        orders=orderlist, 
        fscs=fsclist, 
        fobj=fobj,
        )

    # validate the pg against fscfull
    emdalogger.log_newline(fobj)
    a = np.zeros((len(emmap1.symdat), len(emmap1.res_arr)))
    for n in range(len(emmap1.symdat)):
        a[n,:] = emmap1.symdat[n][2][:]
        l = 'ax'+str(n+1) + ': ' + vec2string(emmap1.symdat[n][1]) + ' Order: ' + str(emmap1.symdat[n][0][0])
        print(l)
        fobj.write(l+'\n')
    m = len(emmap1.symdat) * 6 + len(emmap1.symdat) - 1
    nspaces = 6 + len(emmap1.symdat) * 6 + 6 + 6 + len(emmap1.symdat) - 1
    dash = "-"*nspaces
    print(dash)
    fobj.write(dash+'\n')
    line1 = "  res  | " + " "*((m-4)//2) + "FSCs" + " "*((m-4)//2) + " |  FSCf " 
    print(line1)
    fobj.write(line1+'\n')
    #line2 = "       | " + "-"*m + " |       " 
    line2 = "       | " + "-"*m + " |       "
    print(line2)
    print(dash)
    fobj.write(dash+'\n')
    for i, res in enumerate(emmap1.res_arr):
        print("{:>6.2f} | {} | {:>6.3f}".format(res, vec2string(a[:,i]), emmap1.fscfull[i]))
        fobj.write("{:>6.2f} | {} | {:>6.3f}\n".format(res, vec2string(a[:,i]), emmap1.fscfull[i]))
    fobj.write(dash+'\n')

    # average symcopies
    print()
    print('********** Average symmetry copies ************')
    print('comparison of halfmap-FSC (FSCh) with '
          'symmetry averaged halfmap-FSC per axis (FSCs)')
    fobj.write('********** Average symmetry copies ************\n')
    fobj.write('comparison of halfmap-FSC (FSCh) with '
          'symmetry averaged halfmap-FSC per axis (FSCs)\n')
    axes, folds, tlist = [], [], []
    fsclist = []

    for i, row in enumerate(emmap1.symdat):
        fold, axis, t = row[0], row[1], row[3]
        axes.append(axis)
        folds.append(fold[0])
        fsc_hf, fsc_symhf = avgsym.main(
            f_list=[fhf1, fhf2], 
            axes=[axis], 
            folds=fold, 
            tlist=[t], 
            bin_idx=emmap1.bin_idx,
            nbin=emmap1.nbin)
        fsclist.append(fsc_symhf)
    # printing fscs
    b = np.zeros((len(axes), len(emmap1.res_arr)), 'float')
    for i in range(len(axes)):
        b[i,:] = fsclist[i]
        l = 'ax'+str(i+1) + ': ' + vec2string(axes[i]) + ' Order: ' + str(folds[i])
        print(l)
    print(dash)
    fobj.write(dash+'\n')
    line1 = "  res  | " + " "*((m-4)//2) + "FSCs" + " "*((m-4)//2) + " |  FSCh "
    print(line1)
    fobj.write(line1+'\n')
    print(dash)
    fobj.write(dash+'\n')
    for i, res in enumerate(emmap1.res_arr):
        print("{:>6.2f} | {} | {:>6.3f}".format(res, vec2string(b[:,i]), fsc_hf[i]))
        fobj.write("{:>6.2f} | {} | {:>6.3f}\n".format(res, vec2string(b[:,i]), fsc_hf[i]))
    fobj.write(dash+'\n')

    # FSC levels and point groups
    print()
    print("****** Pointgroup at different resolutions ******")
    fobj.write("****** Pointgroup at different resolutions ******\n")
    levels = [0.95, 0.90, 0.85, 0.80]
    pglevels = []
    for level in levels:
        pglevels.append(get_pg_perlevel(a, axes, folds, level))
    print("  Resol.  pg@lvls= [ 0.95,  0.90,  0.85  0.80 ]")
    fobj.write(" Resol.  pg@lvls= [ 0.95,  0.90,  0.85  0.80 ]\n")
    for i, res in enumerate(res_arr):
        l = [pglevels[n][i] for n in range(len(pglevels))]
        print(res, list2string(l))
        fobj.write(" %.2f  %s\n" %(res, list2string(l)))
    return pg


def get_pg_perlevel(a, axes, folds, level):
    pglist = []
    #print('levle=', level)
    for i in range(a.shape[1]):
        mask = (a[:,i] >= level)
        axlist = []
        orderlist = []
        for j, ax in enumerate(axes):
            if mask[j]:
                axlist.append(ax)
                orderlist.append(folds[j])
        pg = decide_pointgroup(axlist, orderlist)[0]
        #print(i, mask, level, pg)
        pglist.append(pg)    
    return pglist


def main(half1, resol=None, fobj=None, imask=None):
    if resol is not None:
        resol = resol * 1.1 # taking 10% less resolution of author claimed
    # open halfmaps
    half2 = half1.replace("map_1", "map_2")
    print(half1)
    print(half2)
    h1 = iotools.Map(half1)
    h1.read()
    h2 = iotools.Map(half2)
    h2.read()    

    m = re.search('emd_(.+)_half', half1)
    logname = 'emd-%s-pointgroup.txt'%m.group(1)
    if fobj is None:
        fobj = open(logname, 'w')
    maskname = 'emda_mapmask_emd-'+m.group(1)+'.mrc'
    reboxedmaskname = 'emda_rbxmapmask_emd-'+m.group(1)+'.mrc'
    reboxedmapname = 'emda_rbxfullmap_emd-'+m.group(1)+'.mrc'

    # get the mask for future calculations
    if imask is None:
        # calculate EMDA mask from map
        mask_from_halfmaps(half1=half1, maskname=maskname)
        mm = iotools.Map(name=maskname)
        mm.read()
    else:
        mm = iotools.Map(name=imask)
        mm.read()
        
    # reboxing halfmaps using the mask
    print('Reboxing...')
    rmap1, rmask = em.rebox_by_mask(arr=h1.workarr, mask=mm.workarr, mask_origin=mm.origin)
    rmap2, rmask = em.rebox_by_mask(arr=h2.workarr, mask=mm.workarr, mask_origin=mm.origin)
    fullmap = (rmap1 + rmap2) / 2
    # write out reboxed fullmap and mask
    newcell = [fullmap.shape[i]*h1.workcell[i]/shp for i, shp in enumerate(h1.workarr.shape)]
    writemap(fullmap, newcell, reboxedmapname)
    writemap(rmask, newcell, reboxedmaskname)

    # create resolution grid
    for _ in range(3): newcell.append(90.0)
    nbin, res_arr, bin_idx, sgrid = em.get_binidx(cell=newcell, arr=rmap1)
    binfsc = em.fsc(
        f1=fftshift(fftn(rmap1 * rmask)), 
        f2=fftshift(fftn(rmap2 * rmask)), 
        bin_idx=bin_idx, 
        nbin=nbin
        )
    fsc_full = 2 * binfsc / (1. + binfsc)
    # if resol is not given, find it using FSC
    if resol is None:
        ibin = get_ibin(filter_fsc(binfsc), cutoff=0.15)
        resol = res_arr[ibin]
        print(binfsc)
        print('resol: ', resol)
    
    # running proshade
    print('Running Proshade...')
    results = proshade_tools.get_symmops_from_proshade(mapname=reboxedmapname, fobj=fobj)
    if len(results) == 7:
        proshade_pg = results[-2]
        axlist, orderlist, fsclist = proshade_tools.process_proshade_results(results)
        # find point group
        m2 = iotools.Map(name=reboxedmapname)
        m2.read()
        m2.workarr = m2.workarr * rmask
        emda_pg = get_pg(
            axlist=axlist, 
            orderlist=orderlist, 
            fsclist=fsclist, 
            m1=m2, 
            claimed_res=resol, 
            fobj=fobj,
            bin_idx=bin_idx,
            nbin=nbin,
            sgrid=sgrid,
            fsc_full=fsc_full,
            res_arr=res_arr,
            mask=rmask,
            half1=rmap1,
            half2=rmap2,
            )
        print('Proshade PG: ', proshade_pg)
        print('EMDA PG: ', emda_pg)
        strng = "Point group [Proshade, EMDA]: {} {}\n".format(proshade_pg, emda_pg)
        fobj.write(strng)
        return [proshade_pg, emda_pg, maskname, reboxedmapname]
    else:
        return [maskname, reboxedmapname]


def getlist(fname="ids.txt", search_key='EMD-(.+)'):
    fid = open(fname, 'r')
    Lines = fid.read().splitlines()
    emdbidList = []
    for line in Lines:
        m = re.search(search_key, line)
        if m is not None:
            emdbidList.append(m.group(1))
    return emdbidList


def my_func(emdbid):
    try:
        results = fetch_halfmaps(emdbid)
        if len(results) > 0:
            name_list, resol, pg, maskfile = results
            if float(resol) < 10.0 and (pg is not None):
                half1, half2 = name_list[0], name_list[1]            
                filename = "pg_log_emd-%s.txt" %emdbid
                logfile = open(filename, "w")
                logfile.write("\nEMD-{} {} {}\n".format(emdbid, resol, pg))
                final_results = main(
                    half1=half1, resol=float(resol), fobj=logfile, imask=maskfile)
                if len(final_results) == 4:
                    proshade_pg = final_results[0]
                    emda_pg = final_results[1]
                    maskname = final_results[2]
                    reboxmapname = final_results[3]
                    if proshade_pg == '0' : proshade_pg = 'C1'
                    if proshade_pg == 'I0' : proshade_pg = 'I'
                    if proshade_pg == 'O0' : proshade_pg = 'O'
                    if emda_pg == 'None' : emda_pg = 'C1'
                    logfile.write("emda pointgroup: %s\n" %emda_pg)
                    strng = "xEMD-{} {} {} {} {}\n".format(emdbid, resol, pg, proshade_pg, emda_pg)
                    logfile.write(strng)
                    #os.remove(half1)
                    #os.remove(half2)
                    os.remove(maskname)
                    os.remove(reboxmapname)
                    #os.remove(maskfile)
                    logfile.close()
                    print('Structure %s done!' % emdbid)
                else:
                    maskname = final_results[0]
                    reboxmapname = final_results[1]
                    #os.remove(half1)
                    #os.remove(half2)
                    #os.remove(maskfile)
                    os.remove(maskname)
                    os.remove(reboxmapname)
                os.remove(half1)
                os.remove(half2)
                os.remove(maskfile)                
            else:
                os.remove(half1)
                os.remove(half2)    
                os.remove(maskfile)            
        else:
            print('Empty results from xml_read!')
    except:
        pass


if __name__=="__main__":
    #halfmap1 = "/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3651/other/emd_3651_half_map_1.map"
    #halfmap1 = "/Users/ranganaw/MRC/REFMAC/EMD-6952/other/test_symmetry/emd_6952_half_map_1.map"
    #halfmap1 = "/Users/ranganaw/MRC/REFMAC/Vinoth/test_sym/emd_0000_half_map_1.map"
    #halfmap1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-11270/emd_11270_half_map_1.map"
    #halfmap1 = "/Users/ranganaw/MRC/REFMAC/beta_galactosidase/EMD-0153/emd_0153_half_map_1.map"
    #halfmap1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-9191/emd_9191_half_map_1.map"
    #halfmap1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-3693/emd_3693_half_map_1.map"
    #halfmap1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-10561/EMD-10561/emd_10561_half_map_1.map"
    #halfmap1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-0273/emd_0273_half_map_1_resampled.map"

    halfmaplist = [
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-6952/emd_6952_half_map_1.map", #C3
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10561/emd_10561_half_map_1.map", #C35
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0153/emd_0153_half_map_1.map", #D2
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-25437/emd_25437_half_map_1.map", #T
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0263/emd_0263_half_map_1.map", #O
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12181/emd_12181_half_map_1.map", #I
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-12139/reboxed_emd_12139_half_map_1.mrc",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-23884/emd_23884_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21185/emd_21185_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10461/emd_10461_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-20690/emd_20690_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-21246/emd_21246_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-23610/emd_23610_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-23884/emd_23884_half_map_1.map",
        "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21231/emd_21231_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-4906/emd_4906_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-11220/emd_11220_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12073/emd_12073_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-12608/emd_12608_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12819/emd_12819_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12707/emd_12707_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12644/emd_12644_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-13624/emd_13624_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-13844/emd_13844_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-7446/emd_7446_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-9953/emd_9953_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0121/emd_0121_half_map_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10325/emd_10325_half_map_1.map"
    ]
    
    masklist = [
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-6952/emd_6952_msk_1.map"#manual_emda_mask_0.mrc",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10561/emda_mapmask.mrc",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0153/emd_0153_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0263/emd_0263_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12181/manual_emda_mask_0.mrc",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-12139/emda_reboxedmask.mrc",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-23884/emd_23884_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21185/emd_21185_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10461/emd_10461_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-20690/emd_20690_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-21246/manual_emda_mask_0.mrc",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-23610/manual_emda_mask_0.mrc",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-23884/manual_emda_mask_0.mrc",
        "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21231/emd_21231_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-4906/manual_emda_mask_0.mrc"#emd_4906_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-11220/emd_11220_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12073/emd_12073_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-12608/emd_12608_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12819/manual_emda_mask_0.mrc",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12707/manual_emda_mask_0.mrc",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12644/emd_12644_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-13624/emd_13624_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-13844/emd_13844_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/EMD-7446/emd_7446_msk_1.map" #manual_emda_mask_0.mrc",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-9953/emd_9953_msk_1.map",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0121/emd_0121_msk_1.map"#manual_emda_mask_0.mrc",
        #"/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10325/emd_10325_msk_1.map"
    ]

    for ihalf, imask in zip(halfmaplist, masklist):
        main(half1=ihalf, imask=imask, resol=float(4.6))
    exit()

    emdlist1 = getlist(fname="failed_ids.txt")
    #emdbidList = list(set(emdlist1) ^ set(emdlist2)) # XOR operation
    emdbidList = emdlist1
    # parellel processing
    import multiprocessing as mp
    #pool = mp.Pool(mp.cpu_count())
    pool = mp.Pool(20) # use only 1 CPUs for now
    result = pool.map(my_func, emdbidList)

    # cleaning
    pool.close()
    pool.join()