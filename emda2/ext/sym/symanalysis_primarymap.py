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

# from emda2.ext.mapmask import mask_from_halfmaps
from emda2.ext.sym import proshade_tools
from emda2.ext.utils import (
    # get_ibin,
    # filter_fsc,
    # shift_density,
    center_of_mass_density,
    # lowpassmap_butterworth,
)
import fcodes2
from emda2.ext.sym import axis_refinement

# from more_itertools import sort_together
# from emda2.ext.sym import pgcode
# from emda2.ext.sym.download_halfmaps import fetch_halfmaps
from emda2.core import emdalogger
from emda2.core.emdalogger import vec2string, list2string

# from emda2.ext.sym.decide_pointgroup import decide_pointgroup
# import emda2.ext.sym.average_symcopies as avgsym
from emda2.ext.sym.symanalysis_pipeline import (
    _lowpassmap_butterworth,
    test,
    get_pg_perlevel,
    writemap,
)
from emda2.ext.sym.download_halfmaps import fetch_primarymap


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
    # half2 = dict["rmap2"]
    claimed_res = dict["resol"]
    resol4refinement = dict["resol4refinement"]
    output_maps = dict["output_maps"]
    # symaverage = dict["symaverage"]
    label = dict["label"]
    fitfsc = dict["fitfsc"]
    ncycles = dict["ncycles"]
    # pg_decide_fsc = dict["pg_decide_fsc"]

    pix = [dict["newcell"][i] / sh for i, sh in enumerate(half1.shape)]

    emdalogger.log_string(
        fobj,
        "Claimed resolution=%.3f (A)" % claimed_res,
    )
    considered_res = (
        claimed_res * 1.1
    )  # taking 10% less resolution of author claimed
    cbin = np.argmin(np.abs(res_arr - considered_res))
    if res_arr[cbin] <= considered_res:
        cbin -= 1
    considered_res = res_arr[cbin]
    emdalogger.log_string(
        fobj,
        "Considered resolution=%.3f (A), cbin=%i " % (considered_res, cbin),
    )

    if resol4refinement < considered_res:
        resol4refinement = considered_res
    emdalogger.log_string(
        fobj, "Resolution for refinement=%.3f " % resol4refinement
    )

    # select data upto claimed resol
    lowpass_res = claimed_res  # * 0.9
    emdalogger.log_string(fobj, "Lowpass flitering data to %.3f" % lowpass_res)
    fo = _lowpassmap_butterworth(
        fclist=[fftshift(fftn(fftshift(half1 * mask)))],
        sgrid=sgrid,
        smax=lowpass_res,
    )[0]

    # calculate halfmap FSC
    # binfsc = em.fsc(f1=fhf1, f2=fhf2, bin_idx=bin_idx, nbin=nbin)
    binfsc = np.zeros(nbin, "float")
    fsc_full = np.zeros(nbin, "float")
    """ fsc_full = 2 * binfsc / (1.0 + binfsc)  # fullmap FSC
    emdalogger.log_string(fobj, "Halfmap and Fullmap FSCs")
    emdalogger.log_fsc(
        fobj, {"Resol.": res_arr, "FSC(half)": binfsc, "FSC(full)": fsc_full}
    ) """

    map1 = np.real(ifftshift(ifftn(ifftshift(fo)))) * mask
    nx, ny, nz = map1.shape
    com = center_of_mass_density(map1)
    box_centr = (nx // 2, ny // 2, nz // 2)

    # applying translation in Fourier space (faster)
    t = [-(box_centr[i] - com[i]) / sh for i, sh in enumerate(map1.shape)]
    st = fcodes2.get_st(nx, ny, nz, t)[0]
    fo = st * fo
    com = box_centr

    # Normalizing fullmap FCs
    """ eo = fcodes2.get_normalized_sf_singlemap(
        fo=fo,
        bin_idx=bin_idx,
        nbin=nbin,
        mode=0,
        nx=nx,
        ny=ny,
        nz=nz,
    )
    fsc_max = np.sqrt(filter_fsc(fsc_full))
    eo = eo * fcodes2.read_into_grid(bin_idx, fsc_max, nbin, nx, ny, nz) """

    eo = fo

    # Preparing object for axis refinement
    emmap1 = axis_refinement.EmmapOverlay(arr=map1)
    emmap1.emdbid = label
    emmap1.bin_idx = bin_idx
    emmap1.res_arr = res_arr
    emmap1.nbin = nbin
    emmap1.claimed_res = considered_res  # claimed_res
    emmap1.claimed_bin = cbin  # claimed_cbin
    # emmap1.mask = mask_com_adjusted
    emmap1.pix = pix
    emmap1.fo_lst = [fo]
    emmap1.eo_lst = [eo]
    emmap1.fitres = resol4refinement
    emmap1.fitfsc = fitfsc
    emmap1.ncycles = ncycles
    emmap1.fscfull = fsc_full
    emmap1.fscstar = fsc_full
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
        """ if symaverage:
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
                fobj.write(dash + "\n") """
    return pg


def main(dict, fobj=None):
    # check if primary map is present
    if not (os.path.isfile(dict["pmap"]) and (os.path.isfile(dict["pmap"]))):
        raise SystemExit("Primary map is missing!")

    # make a label
    if dict["label"] is None:
        pattern = r'emd_(\d+).m'
        if re.search(pattern, dict["pmap"]):
            label = re.findall("\d+", dict["pmap"])[-1]
        else:
            label = "0000"
        dict["label"] = label
    else:
        label = dict["label"]

    # file names
    logname = "emd-%s-pointgroup.txt" % label
    emdbid = "emd-%s" % label
    # maskname = "emda_mapmask_emd-%s.mrc" % label
    reboxedmapname = "emda_rbxfullmap_emd-%s.mrc" % label
    dict["emdbid"] = emdbid

    if fobj is None:
        fobj = open(logname, "w")

    """ if dict["resol"] is not None:
        dict["resol"] = (
            dict["resol"] * 1.1
        )  # taking 10% less resolution of author claimed """

    # get the mask
    if dict["mask"] is None:
        raise SystemExit("Please include a mask")
    else:
        mm = iotools.Map(name=dict["mask"])
        mm.read()

    # reading primary map
    m1 = iotools.Map(dict["pmap"])
    m1.read()

    # reboxing primary map using the mask
    print("Reboxing...")
    rmap1, rmask = em.rebox_by_mask(
        arr=m1.workarr, mask=mm.workarr, mask_origin=mm.origin
    )
    dict["rmap1"] = rmap1
    dict["rmask"] = rmask
    fullmap = rmap1 # m1.workarr
    dict["fullmap"] = fullmap

    newcell = [
        fullmap.shape[i] * m1.workcell[i] / shp
        for i, shp in enumerate(m1.workarr.shape)
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
    # if axeslist and orderlist present, don't run proshade
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
        else:
            dict["emda_pg"] = "C1"
            dict["proshade_pg"] = "C1"
            strng = "Point group [Proshade, EMDA]: {} {}\n".format(
                dict["proshade_pg"], dict["emda_pg"]
            )
            fobj.write(strng)
    return dict


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
        "fitfsc": 0.1,
        "lowres_cutoff": 10.0,
        "pg_decide_fsc": 0.9,
        "ncycles": 10,
    }
    try:
        results = fetch_primarymap(emdbid)
        if len(results) > 0:
            name_list, resol, pg, maskfile = results
            params["pmap"] = name_list[0]
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

            if final_results["pmap"] is not None:
                os.remove(final_results["pmap"])
                #os.remove(final_results["mask"])
        else:
            print("Empty results from xml_read!")
    except ValueError:
        pass