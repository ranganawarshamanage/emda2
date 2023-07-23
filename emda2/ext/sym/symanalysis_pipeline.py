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
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from emda2.core import iotools, plotter, fsctools
import emda2.emda_methods2 as em
from emda2.ext.mapmask import mask_from_halfmaps
from emda2.ext.sym import proshade_tools
from emda2.ext.utils import (
    get_ibin,
    filter_fsc,
    shift_density,
    center_of_mass_density,
    lowpassmap_butterworth,
)
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


def _lowpassmap_butterworth(fclist, sgrid, smax):
    order = 4  # order of the butterworth filter
    D = sgrid
    d = 1.0 / smax  # smax in Ansgtrom units
    # butterworth filter
    bwfilter = 1.0 / np.sqrt(1 + ((D / d) ** (2 * order)))
    lw_fclist = []
    for fc in fclist:
        lw_fclist.append(fc * bwfilter)
    return lw_fclist


def is_prime(n):
    """
    https://stackoverflow.com/questions/15285534/isprime-function-for-python-language
    """
    if n == 2 or n == 3:
        return True
    if n < 2 or n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    r = int(n**0.5)
    f = 5
    while f <= r:
        if n % f == 0:
            return False
        if n % (f + 2) == 0:
            return False
        f += 6
    return True


def normalise_axis(axis):
    ax = np.asarray(axis, "float")
    return ax / math.sqrt(np.dot(ax, ax))


def test(emmap1, axes, orders, fscs, fobj=None):
    if fobj is None:
        fobj = open("pointgroup.txt", "w")
    ang_tol_p = 3.0  # angle tolerence for proshade axes
    fsc_thresh_for_axrefine = 0.95  # FCS threshold for axis refinement
    pg_decide_fsc = 0.9  # FSC threshold for pointgroup decision
    s = (
        "\n *** Default parameters used ***\n"
        "Angle tolerence for Proshade axes:     %s (deg)\n"
        "FCS threshold for axis refinement:     %s\n"
        "FSC threshold for pointgroup decision: %s"
        % (ang_tol_p, fsc_thresh_for_axrefine, pg_decide_fsc)
    )
    emdalogger.log_string(fobj, s)
    # choose only prime order axes with +ve FSCs
    prime_orderlist = []
    prime_axlist = []
    prime_fsclist = []
    fsc_tmp_list = []
    for i, order in enumerate(orders):
        if is_prime(order):
            fsc_tmp_list.append(fscs[i])
    s = (
        "\n    *** Chosen prime axes ***"
        "\n--------------------------------------\n"
        " #      X      Y      Z   Order   FSC\n"
        "--------------------------------------"
    )
    emdalogger.log_string(fobj, s)

    for i, order in enumerate(orders):
        if is_prime(order) and fscs[i] > 0.0:
            prime_orderlist.append(order)
            ax = normalise_axis(axes[i])
            prime_axlist.append(ax)
            prime_fsclist.append(fscs[i])
            emdalogger.log_string(
                fobj,
                "%i   %s   %s  % .3f"
                % (i, vec2string(ax), int(order), fscs[i]),
            )
    emdalogger.log_string(fobj, "--------------------------------------")

    if len(prime_orderlist) == 0:
        pg = "C1"
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

    # sort by FSC
    if len(order5ax) > 1:
        sorder5fsc, sorder5ax = sort_together(
            [order5fsc, order5ax], reverse=True
        )
    else:
        sorder5fsc, sorder5ax = order5fsc, order5ax

    if len(order3ax) > 1:
        sorder3fsc, sorder3ax = sort_together(
            [order3fsc, order3ax], reverse=True
        )
    else:
        sorder3fsc, sorder3ax = order3fsc, order3ax

    if len(order2ax) > 1:
        sorder2fsc, sorder2ax = sort_together(
            [order2fsc, order2ax], reverse=True
        )
    else:
        sorder2fsc, sorder2ax = order2fsc, order2ax

    if len(ordernax) > 1:
        sordernfsc, sordernax, sordern = sort_together(
            [ordernfsc, ordernax, ordern], reverse=True
        )
    else:
        sordernfsc, sordernax, sordern = ordernfsc, ordernax, ordern

    s = (
        "\n"
        "Number of 5-fold axes: %s\n"
        "Number of 3-fold axes: %s\n"
        "Number of 2-fold axes: %s\n"
        "Number of n-fold axes: %s\n"
        # "\n        Detecting Pointgroup        \n"
        # "======================================"
        % (len(sorder5ax), len(sorder3ax), len(sorder2ax), len(sordernax))
    )
    emdalogger.log_string(fobj, s)

    print('Checking for cyclic orders...')
    pg = pgcode.check_for_cyclic_only(
        emmap1=emmap1,
        axes=prime_axlist,
        orders=prime_orderlist,
        fscs=prime_fsclist,
        fobj=fobj,
    )
    if pg is not None:
        return pg
    print("More than one axis.")
    pg = "C1"
    if len(sorder5ax) > 0:
        axes = [sorder5ax, sorder3ax, sorder2ax, sordernax]
        fscs = [sorder5fsc, sorder3fsc, sorder2fsc, sordernfsc]
        pg = pgcode.five_folds(
            emmap1=emmap1, axes=axes, fscs=fscs, sordern=sordern, fobj=fobj
        )
    else:
        if len(sorder3ax) > 0:
            axes = [sorder3ax, sorder2ax, sordernax]
            fscs = [sorder3fsc, sorder2fsc, sordernfsc]
            pg = pgcode.no5folds(
                emmap1=emmap1, axes=axes, fscs=fscs, sordern=sordern, fobj=fobj
            )
        else:
            if len(sordernax) > 0:
                axes = [sorder2ax, sordernax]
                fscs = [sorder2fsc, sordernfsc]
                pg = pgcode.no53folds(
                    emmap1=emmap1,
                    axes=axes,
                    fscs=fscs,
                    sordern=sordern,
                    fobj=fobj,
                )
            else:
                pg = pgcode.just2folds(
                    emmap1=emmap1,
                    sorder2ax=sorder2ax,
                    sorder2fsc=sorder2fsc,
                    fobj=fobj,
                )
    return pg


def get_pg(dict, fobj):
    axlist = dict["axlist"]
    orderlist = dict["orderlist"]
    fsclist = dict["fsclist"]
    res_arr = dict["res_arr"]
    bin_idx = dict["bin_idx"]
    nbin = dict["nbin"]
    sgrid = dict["sgrid"]
    # fobj = kwargs["fobj"]
    mask = dict["rmask"]
    half1 = dict["rmap1"]
    half2 = dict["rmap2"]
    claimed_res = dict["resol"]
    resol4refinement = dict["resol4refinement"]
    output_maps = dict["output_maps"]
    symaverage = dict["symaverage"]
    label = dict["label"]
    fitfsc = dict["fitfsc"]
    ncycles = dict["ncycles"]
    pg_decide_fsc = dict["pg_decide_fsc"]

    pix = [dict["newcell"][i] / sh for i, sh in enumerate(half1.shape)]

    emdalogger.log_string(
        fobj,
        "Claimed resolution=%.3f (A)" % claimed_res,
    )
    claimed_cbin = np.argmin(np.abs(res_arr - claimed_res))
    if res_arr[claimed_cbin] <= claimed_res:
        claimed_cbin -= 1
    claimed_res = res_arr[claimed_cbin]
    emdalogger.log_string(
        fobj,
        "Considered resolution=%.3f (A), cbin=%i "
        % (claimed_res, claimed_cbin),
    )

    if resol4refinement < claimed_res:
        resol4refinement = claimed_res
    emdalogger.log_string(
        fobj, "Resolution for refinement=%.3f " % resol4refinement
    )

    # select data upto claimed resol
    lowpass_res = claimed_res * 0.9
    emdalogger.log_string(fobj, "Lowpass flitering data to %.3f" % lowpass_res)
    fhf1, fhf2 = _lowpassmap_butterworth(
        fclist=[
            fftshift(fftn(fftshift(half1 * mask))),
            fftshift(fftn(fftshift(half2 * mask))),
        ],
        sgrid=sgrid,
        smax=lowpass_res,
    )
    fo = (fhf1 + fhf2) / 2

    # calculate halfmap FSC
    binfsc = em.fsc(f1=fhf1, f2=fhf2, bin_idx=bin_idx, nbin=nbin)
    fsc_full = 2 * binfsc / (1.0 + binfsc)  # fullmap FSC
    emdalogger.log_string(fobj, "Halfmap and Fullmap FSCs")
    emdalogger.log_fsc(
        fobj, {"Resol.": res_arr, "FSC(half)": binfsc, "FSC(full)": fsc_full}
    )

    map1 = np.real(ifftshift(ifftn(ifftshift(fo)))) * mask
    nx, ny, nz = map1.shape
    com = center_of_mass_density(map1)
    box_centr = (nx // 2, ny // 2, nz // 2)

    # applying translation in Fourier space (faster)
    t = [-(box_centr[i] - com[i]) / sh for i, sh in enumerate(map1.shape)]
    st = fcodes2.get_st(nx, ny, nz, t)[0]
    fhf1 = st * fhf1
    fhf2 = st * fhf2
    fo = st * fo
    com = box_centr

    # Normalizing fullmap FCs
    eo = fcodes2.get_normalized_sf_singlemap(
        fo=fo,
        bin_idx=bin_idx,
        nbin=nbin,
        mode=0,
        nx=nx,
        ny=ny,
        nz=nz,
    )
    fsc_max = np.sqrt(filter_fsc(fsc_full))
    eo = eo * fcodes2.read_into_grid(bin_idx, fsc_max, nbin, nx, ny, nz)

    # output normalized and weighted map
    if output_maps:
        bm = iotools.Map("nem_%s.mrc" % label)
        bm.arr = np.real(ifftshift(ifftn(ifftshift(eo))))
        bm.cell = dict["newcell"]
        # bm.axorder = m1.axorder
        bm.origin = [0, 0, 0]
        bm.write()

    # Preparing object for axis refinement
    emmap1 = axis_refinement.EmmapOverlay(arr=map1)
    emmap1.emdbid = label
    emmap1.bin_idx = bin_idx
    emmap1.res_arr = res_arr
    emmap1.nbin = nbin
    emmap1.claimed_res = claimed_res
    emmap1.claimed_bin = claimed_cbin
    # emmap1.mask = mask_com_adjusted
    emmap1.pix = pix
    emmap1.fo_lst = [fo]
    emmap1.eo_lst = [eo]
    emmap1.fitres = resol4refinement
    emmap1.fitfsc = fitfsc
    emmap1.ncycles = ncycles
    emmap1.fscfull = fsc_full
    emmap1.fscstar = fsc_max
    emmap1.symdat = []
    emmap1.syminfo = []
    emmap1.com = True
    emmap1.com1 = com
    emmap1.map_dim = eo.shape
    emmap1.map_unit_cell = dict["newcell"]
    emmap1.output_maps = output_maps  # bool
    emdalogger.log_vector(
        fobj, {"\nCentre of Mass [x, y, z] (pixel units)": list(com)}
    )
    emdalogger.log_vector(
        fobj,
        {
            "Centre of Mass [x, y, z] (A)": [
                com[i] * emmap1.pix[i] for i in range(3)
            ]
        },
    )

    emdalogger.log_string(fobj, "\nFinding pointgroup...")
    pg = test(
        emmap1=emmap1,
        axes=axlist,
        orders=orderlist,
        fscs=fsclist,
        fobj=fobj,
    )

    if len(emmap1.symdat) > 0:
        symdat = emmap1.symdat
        # validate the pg against fscfull
        emdalogger.log_newline(fobj)
        a = np.zeros((len(symdat), len(emmap1.res_arr)))
        for n in range(len(symdat)):
            a[n, :] = symdat[n][2][:]
            emdalogger.log_string(
                fobj,
                (
                    f"ax{n+1}: {vec2string(symdat[n][1])} Order:"
                    f" {str(symdat[n][0][0])}"
                ),
            )

        m = len(symdat) * 6 + len(symdat) - 1
        nspaces = 6 + len(symdat) * 6 + 6 + 6 + len(symdat) - 1
        dash = "-" * nspaces
        emdalogger.log_string(fobj, dash)
        emdalogger.log_string(
            fobj,
            (
                f"  res  | {' ' * ((m - 4) // 2)}FSCs{' ' * ((m - 4) // 2)} | "
                " FSCf "
            ),
        )
        emdalogger.log_string(fobj, f"       | {'-' * m} |       ")
        for i, res in enumerate(emmap1.res_arr):
            emdalogger.log_string(
                fobj,
                (
                    f"{res:>6.2f} | {vec2string(a[:, i])} |"
                    f" {emmap1.fscfull[i]:>6.3f}"
                ),
            )
            if i == emmap1.claimed_bin:
                emdalogger.log_string(fobj, dash)
        fobj.write(dash + "\n")

        # FSC levels and point groups
        axes, folds = [], []
        fsclist = []
        for i, row in enumerate(symdat):
            fold, axis = row[0], row[1]
            folds.append(fold[0])
            axes.append(axis)
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
            ll = [pglevels[n][i] for n in range(len(pglevels))]
            print(res, list2string(ll))
            fobj.write(" %.2f  %s\n" % (res, list2string(ll)))
            if i == emmap1.claimed_bin:
                print(dash)
                fobj.write(dash + "\n")

        # average symcopies
        if symaverage:
            s = (
                "\n********** Average symmetry copies ************\n"
                "Comparison of halfmap-FSC (FSCh) with "
                "symmetry averaged halfmap-FSC per axis (FSCs)\n"
            )
            emdalogger.log_string(fobj, s)
            axes, folds = [], []
            fsclist = []
            for i, row in enumerate(symdat):
                fold, axis, binfsc, t = row
                axes.append(axis)
                folds.append(fold[0])
                print("computing FSC between sym. averaged halves ....")
                fsc_hf = fsctools.anytwomaps_fsc_covariance(
                    fhf1, fhf2, bin_idx, nbin
                )[0]
                favghf = avgsym.symavg_peraxis_perorder(
                    f_list=[fhf1, fhf2],
                    axes=[axis],
                    folds=fold,
                    tlist=[t],
                    bin_idx=emmap1.bin_idx,
                    nbin=emmap1.nbin,
                )
                fsc_symhf = fsctools.anytwomaps_fsc_covariance(
                    favghf[0], favghf[1], bin_idx, nbin
                )[0]
                # output symmetry averaged maps
                if emmap1.output_maps:
                    mapname1 = "%s_avghf1_ax%s_fold%s.mrc" % (
                        emmap1.emdbid,
                        i,
                        fold[0],
                    )
                    mapname2 = "%s_avghf2_ax%s_fold%s.mrc" % (
                        emmap1.emdbid,
                        i,
                        fold[0],
                    )
                    h1out = iotools.Map(mapname1)
                    h1out.arr = np.real(ifftshift(ifftn(ifftshift(favghf[0]))))
                    h1out.cell = emmap1.map_unit_cell
                    h1out.origin = [0, 0, 0]
                    h1out.write()
                    h2out = iotools.Map(mapname2)
                    h2out.arr = np.real(ifftshift(ifftn(ifftshift(favghf[1]))))
                    h2out.cell = emmap1.map_unit_cell
                    h2out.origin = [0, 0, 0]
                    h2out.write()
                    fsclist.append(fsc_symhf)

            # printing fscs
            if len(axes) > 0:
                b = np.zeros((len(axes), len(emmap1.res_arr)), "float")
                for i in range(len(axes)):
                    b[i, :] = fsclist[i]
                    string1 = (
                        "ax"
                        + str(i + 1)
                        + ": "
                        + vec2string(axes[i])
                        + " Order: "
                        + str(folds[i])
                    )
                    print(string1)
                print(dash)
                fobj.write(dash + "\n")
                line1 = (
                    "  res  | "
                    + " " * ((m - 4) // 2)
                    + "FSCs"
                    + " " * ((m - 4) // 2)
                    + " |  FSCh "
                )
                print(line1)
                fobj.write(line1 + "\n")
                print(dash)
                fobj.write(dash + "\n")
                for i, res in enumerate(emmap1.res_arr):
                    print(
                        "{:>6.2f} | {} | {:>6.3f}".format(
                            res, vec2string(b[:, i]), fsc_hf[i]
                        )
                    )
                    fobj.write(
                        "{:>6.2f} | {} | {:>6.3f}\n".format(
                            res, vec2string(b[:, i]), fsc_hf[i]
                        )
                    )
                    if i == emmap1.claimed_bin:
                        print(dash)
                        fobj.write(dash + "\n")
                fobj.write(dash + "\n")
                print("Plotting FSCs...")
                # plotting FSCs
                labels = ["hf1-hf2"]
                for i in range(len(axes)):
                    labels.append(
                        "Ax%s Or%s:avghf1-avghf2" % (str(i + 1), str(folds[i]))
                    )
                fsclist.insert(0, fsc_hf)
                plotter.plot_nlines(
                    res_arr=emmap1.res_arr,
                    list_arr=fsclist,
                    mapname="emd-%s_emda_halfmap-fscs" % emmap1.emdbid,
                    curve_label=labels,
                    fscline=0.0,
                    plot_title="FSC between half maps",
                )
            else:
                print(
                    "---- None of the axis has FSC_sym >= %s @ %.2f A"
                    % (pg_decide_fsc, emmap1.claimed_res)
                )
                fobj.write(
                    "---- None of the axis has FSC_sym >= %s @ %.2f A\n"
                    % (pg_decide_fsc, emmap1.claimed_res)
                )
                fobj.write(dash + "\n")
    return pg


def get_pg_perlevel(a, axes, folds, level):
    pglist = []
    for i in range(a.shape[1]):
        mask = a[:, i] >= level
        axlist = []
        orderlist = []
        for j, ax in enumerate(axes):
            if mask[j]:
                axlist.append(ax)
                orderlist.append(folds[j])
        pg = decide_pointgroup(axlist, orderlist)[0]
        pglist.append(pg)
    return pglist


def main(dict, fobj=None):
    # check if halfmaps are present
    if not (os.path.isfile(dict["half1"])):
        raise SystemExit("Half maps are missing!")

    # make a label
    if dict["label"] is None:
        pattern = r"emd_\d+_half"
        if re.search(pattern, dict["half1"]):
            label = re.findall("\d+", dict["half1"])[0]
        else:
            label = "0000"
        dict["label"] = label
    else:
        label = dict["label"]

    # file names
    logname = "emd-%s-pointgroup.txt" % label
    emdbid = "emd-%s" % label
    maskname = "emda_mapmask_emd-%s.mrc" % label
    # reboxedmaskname = "emda_rbxmapmask_emd-%s.mrc" % label
    reboxedmapname = "emda_rbxfullmap_emd-%s.mrc" % label
    dict["emdbid"] = emdbid

    if fobj is None:
        fobj = open(logname, "w")

    if dict["resol"] is not None:
        dict["resol"] = (
            dict["resol"] * 1.1
        )  # taking 10% less resolution of author claimed

    # get the mask
    if dict["mask"] is None:
        # calculate EMDA mask from map
        dict["mask"] = maskname
        mask_from_halfmaps(half1=dict["half1"], maskname=maskname)
        mm = iotools.Map(name=maskname)
        mm.read()
    else:
        mm = iotools.Map(name=dict["mask"])
        mm.read()

    # reading half maps
    h1 = iotools.Map(dict["half1"])
    h1.read()
    h2 = iotools.Map(dict["half2"])
    h2.read()

    # reboxing halfmaps using the mask
    print("Reboxing...")
    rmap1, rmask = em.rebox_by_mask(
        arr=h1.workarr, mask=mm.workarr, mask_origin=mm.origin
    )
    dict["rmap1"] = rmap1
    dict["rmask"] = rmask
    rmap2, rmask = em.rebox_by_mask(
        arr=h2.workarr, mask=mm.workarr, mask_origin=mm.origin
    )
    dict["rmap2"] = rmap2
    fullmap = (rmap1 + rmap2) / 2
    dict["fullmap"] = fullmap

    newcell = [
        fullmap.shape[i] * h1.workcell[i] / shp
        for i, shp in enumerate(h1.workarr.shape)
    ]
    for _ in range(3):
        newcell.append(90.0)
    dict["newcell"] = newcell

    # create resolution grid
    nbin, res_arr, bin_idx, sgrid = em.get_binidx(cell=newcell, arr=rmap1)
    dict["nbin"] = nbin
    dict["res_arr"] = res_arr
    dict["bin_idx"] = bin_idx
    dict["sgrid"] = sgrid
    # if axeslist and orderlist present don't run proshade
    if dict["axlist"] is not None and dict["orderlist"] is not None:
        assert len(dict["axlist"]) == len(dict["orderlist"])
        if dict["fsclist"] is None:
            dict["fsclist"] = [0.1 for _ in range(len(dict["axlist"]))]
        # find point group
        emda_pg = get_pg(dict, fobj)
        dict["emda_pg"] = emda_pg
    else:
        # write out reboxed fullmap for proshade
        writemap(fullmap, newcell, reboxedmapname)
        dict["reboxedmapname"] = reboxedmapname
        # run proshade
        print("Running Proshade...")
        results = proshade_tools.get_symmops_from_proshade(
            mapname=reboxedmapname, fobj=fobj
        )
        if results is not None and len(results) == 7:
            proshade_pg = results[-2]
            (
                axlist,
                orderlist,
                fsclist,
            ) = proshade_tools.process_proshade_results(results)
            dict["axlist"] = axlist
            dict["orderlist"] = orderlist
            dict["fsclist"] = fsclist
            # find point group
            emda_pg = get_pg(dict, fobj)
            dict["emda_pg"] = emda_pg
            dict["proshade_pg"] = proshade_pg
            strng = "Point group [Proshade, EMDA]: {} {}\n".format(
                proshade_pg, emda_pg
            )
            fobj.write(strng)
    return dict


def switch(dict):
    print(dict)
    if (os.path.isfile(dict["half1"])):
        return main(dict, fobj=None)
    if (os.path.isfile(dict["pmap"])):
        from emda2.ext.sym import symanalysis_primarymap
        return symanalysis_primarymap.main(dict, fobj=None)
    else:
        print("Maps are missing!")
        return None


def getlist(fname="ids.txt", search_key="EMD-(.+)"):
    fid = open(fname, "r")
    Lines = fid.read().splitlines()
    emdbidList = []
    for line in Lines:
        m = re.search(search_key, line)
        if m is not None:
            emdbidList.append(m.group(1))
    return emdbidList


def my_func(emdbid):
    params = {
        "half1": None,
        "half2": None,
        "mask": None,
        "user_pg": None,
        "resol": None,
        "label": emdbid,
        "resol4refinement": 5.0,
        "output_maps": False,
        "symaverage": False,
        "axlist": None,
        "orderlist": None,
        "fsclist": None,
        "lowres_cutoff": 10.0,
        "pg_decide_fsc": 0.9,
    }
    try:
        results = fetch_halfmaps(emdbid)
        if len(results) > 0:
            name_list, resol, pg, maskfile = results
            params["half1"] = name_list[0]
            params["half2"] = name_list[1]
            params["mask"] = maskfile
            params["resol"] = float(resol)
            params["user_pg"] = pg
            if float(resol) < params["lowres_cutoff"] and (pg is not None):
                filename = "pg_log_emd-%s.txt" % emdbid
                logfile = open(filename, "w")
                logfile.write("\nEMD-{} {} {}\n".format(emdbid, resol, pg))
                final_results = main(params, logfile)
                if (
                    final_results["proshade_pg"] is not None
                    and final_results["emda_pg"] is not None
                ):
                    proshade_pg = final_results["proshade_pg"]
                    emda_pg = final_results["emda_pg"]
                    if proshade_pg == "0":
                        proshade_pg = "C1"
                    if proshade_pg == "I0":
                        proshade_pg = "I"
                    if proshade_pg == "O0":
                        proshade_pg = "O"
                    if emda_pg == "None":
                        emda_pg = "C1"
                    logfile.write("emda pointgroup: %s\n" % emda_pg)
                    strng = "xEMD-{} {} {} {} {}\n".format(
                        emdbid, resol, pg, proshade_pg, emda_pg
                    )
                    logfile.write(strng)
                    logfile.close()
                    print("Structure %s done!" % emdbid)

            if final_results["mask"] is not None:
                os.remove(final_results["mask"])

            if final_results["reboxedmapname"] is not None:
                os.remove(final_results["reboxedmapname"])

            if final_results["half1"] is not None:
                os.remove(final_results["half1"])
                os.remove(final_results["half2"])
                os.remove(final_results["mask"])
        else:
            print("Empty results from xml_read!")
    except ValueError:
        pass


if __name__ == "__main__":
    # halfmap1 = "/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3651/other/emd_3651_half_map_1.map"
    # halfmap1 = "/Users/ranganaw/MRC/REFMAC/EMD-6952/other/test_symmetry/emd_6952_half_map_1.map"
    # halfmap1 = "/Users/ranganaw/MRC/REFMAC/Vinoth/test_sym/emd_0000_half_map_1.map"
    # halfmap1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-11270/emd_11270_half_map_1.map"
    # halfmap1 = "/Users/ranganaw/MRC/REFMAC/beta_galactosidase/EMD-0153/emd_0153_half_map_1.map"
    # halfmap1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-9191/emd_9191_half_map_1.map"
    # halfmap1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-3693/emd_3693_half_map_1.map"
    # halfmap1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-10561/EMD-10561/emd_10561_half_map_1.map"
    # halfmap1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-0273/emd_0273_half_map_1_resampled.map"

    halfmaplist = [
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-6952/emd_6952_half_map_1.map", #C3
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10561/emd_10561_half_map_1.map", #C35
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0153/emd_0153_half_map_1.map", #D2
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-25437/emd_25437_half_map_1.map", #T
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0263/emd_0263_half_map_1.map", #O
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12181/emd_12181_half_map_1.map", #I
        # "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-12139/reboxed_emd_12139_half_map_1.mrc",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-23884/emd_23884_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21185/emd_21185_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10461/emd_10461_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-20690/emd_20690_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-21246/emd_21246_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-23610/emd_23610_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-23884/emd_23884_half_map_1.map",
        "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21231/emd_21231_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-4906/emd_4906_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-11220/emd_11220_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12073/emd_12073_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-12608/emd_12608_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12819/emd_12819_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12707/emd_12707_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12644/emd_12644_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-13624/emd_13624_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-13844/emd_13844_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-7446/emd_7446_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-9953/emd_9953_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0121/emd_0121_half_map_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10325/emd_10325_half_map_1.map"
    ]

    masklist = [
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-6952/emd_6952_msk_1.map"#manual_emda_mask_0.mrc",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10561/emda_mapmask.mrc",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0153/emd_0153_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0263/emd_0263_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12181/manual_emda_mask_0.mrc",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-12139/emda_reboxedmask.mrc",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-23884/emd_23884_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21185/emd_21185_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10461/emd_10461_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-20690/emd_20690_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-21246/manual_emda_mask_0.mrc",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-23610/manual_emda_mask_0.mrc",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-23884/manual_emda_mask_0.mrc",
        "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21231/emd_21231_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-4906/manual_emda_mask_0.mrc"#emd_4906_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-11220/emd_11220_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12073/emd_12073_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-12608/emd_12608_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12819/manual_emda_mask_0.mrc",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12707/manual_emda_mask_0.mrc",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-12644/emd_12644_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-13624/emd_13624_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-13844/emd_13844_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-7446/emd_7446_msk_1.map" #manual_emda_mask_0.mrc",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-9953/emd_9953_msk_1.map",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-0121/emd_0121_msk_1.map"#manual_emda_mask_0.mrc",
        # "/Users/ranganaw/MRC/REFMAC/symmetry/testcases/EMD-10325/emd_10325_msk_1.map"
    ]

    for ihalf, imask in zip(halfmaplist, masklist):
        main(half1=ihalf, imask=imask, resol=float(4.6))
    exit()

    emdlist1 = getlist(fname="failed_ids.txt")
    # emdbidList = list(set(emdlist1) ^ set(emdlist2)) # XOR operation
    emdbidList = emdlist1
    # parellel processing
    import multiprocessing as mp

    # pool = mp.Pool(mp.cpu_count())
    pool = mp.Pool(20)  # use only 1 CPUs for now
    result = pool.map(my_func, emdbidList)

    # cleaning
    pool.close()
    pool.join()
