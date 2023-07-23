"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

import traceback
import numpy as np
import math
from emda2.core import emdalogger
from numpy.fft import fftshift, ifftshift, fftn, ifftn
from emda2.ext.sym.pointgroup import decide_pointgroup
from emda2.core import fsctools, maptools, quaternions
import itertools
from emda2.ext.sym import axis_refinement
import fcodes2
from emda2.ext.utils import rotate_f
from emda2.ext.sym.get_rotation_center import get_t_to_centroid
from more_itertools import sort_together

ang_tol = 1.0
p_resol = 8.0  # resolution for proshade
ang_tol_p = 3.0  # angle tolerence for proshade axes

fsc_thresh_for_axrefine = 0.99  # FCS threshold for axis refinement.
# average FSC upto claimed resolution compared with this

totalang_deg = float(360)  # total angle
pg_decide_fsc = 0.9  # FSC threshold for pointgroup decision
fsc_thresh_for_trueorder = 0.9  # threshold at ? resolution
refined_t = [0.0, 0.0, 0.0]  # default translation
pg = None


def vec2string(vec):
    return " ".join(("% .3f" % x for x in vec))


def cosine_angle(ax1, ax2):
    vec_a = np.asarray(ax1, "float")
    vec_b = np.asarray(ax2, "float")
    dotp = np.dot(vec_a, vec_b)
    if -1.0 <= dotp <= 1.0:
        angle = math.acos(dotp)
    else:
        print("Problem, dotp: ", dotp)
        print("axes:", vec_a, vec_b)
        angle = 0.0
    return np.rad2deg(angle)


def get_rotmat_from_axisangle(axis, angle):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    q = quaternions.get_quaternion(list(axis), angle)
    rotmat = quaternions.get_RM(q)
    return rotmat


def calc_fsc(emmap1, axis, angle, t=None, fobj=None, fo=None):
    if fo is None:
        fo = emmap1.fo_lst[0]
    bin_idx = emmap1.bin_idx
    nbin = emmap1.nbin
    cbin = emmap1.claimed_bin
    print("axis, angle: ", axis, angle)
    if t is None:
        t = np.array([0.0, 0.0, 0.0], "float")
    try:
        print("t: ", t)
        nx, ny, nz = fo.shape
        t = -np.asarray(t, "float")  # reverse the direction
        st = fcodes2.get_st(nx, ny, nz, t)[0]
        rotmat = get_rotmat_from_axisangle(axis, angle)
        print("RM:")
        print(rotmat)
        # NOTE - first translate then rotate.
        fst = fo * st
        frt = rotate_f(rotmat, fst, interp="linear")[:, :, :, 0]
        fsc = fsctools.anytwomaps_fsc_covariance(fst, frt, bin_idx, nbin)[0]
        """ print('~~~~~~~ TEST ~~~~~~~')
        for i, ifsc in enumerate(fsc):
            print(i, emmap1.res_arr[i], ifsc) """
        ax_fsc = fsc[cbin]  # sym. FSC @ claimed resol.
        return [fsc, ax_fsc, frt]
    except:
        if fobj is not None:
            fobj.write(traceback.format_exc())


def is_given_order_correct(emmap1, axis, t, known_order, current_order, fobj):
    K = known_order
    L = current_order
    angledif_list = [
        abs(float(360 / K / 2) - float(360 * j / L)) for j in range(1, L)
    ]
    ang = float(360 * (angledif_list.index(min(angledif_list)) + 1) / L)
    print("corder, angle: ", L, ang)
    fsc_results = calc_fsc(
        emmap1=emmap1,
        axis=axis,
        angle=ang,
        t=t,
    )
    emmap1.symdat.append(
        [[current_order], axis, list(fsc_results[0]), list(t)]
    )
    if fsc_results is not None:
        fsc = fsc_results[1]
        if fsc >= fsc_thresh_for_trueorder:  # thresh needs to improve
            return True
        else:
            return False
    else:
        return False


def get_axorder(emmap1, refined_axis, order_list, fobj=None, t=None):
    emdalogger.log_newline(fobj)
    emdalogger.log_string(fobj, "Finding best order")
    emdalogger.log_string(fobj, "------------------")
    if t is None:
        t = np.array([0.0, 0.0, 0.0], "float")
    order_list = sorted(set(order_list), key=order_list.index)
    print("order_list: ", order_list)
    if len(order_list) > 1:
        K = order_list[0]
        order_mask = [True]
        for order in order_list[1:]:
            response = is_given_order_correct(
                emmap1, refined_axis, t, K, order, fobj
            )
            order_mask.append(response)
        ol = np.asarray(order_list)
        om = np.asarray(order_mask)
        order_list = ol[om]
    try:
        true_order_list = []
        true_fsc_list = []
        print("cleaned order_list: ", order_list)
        for order in order_list:
            current_odr1 = order
            exponent = 1
            # calc FSC for current order
            fsc_results = calc_fsc(
                emmap1=emmap1, axis=refined_axis, angle=float(360 / order), t=t
            )
            if fsc_results is not None:
                fsc = fsc_results[1]
                emdalogger.log_string(
                    fobj,
                    "   Axis: %s   Order: %s   FSC: % .3f"
                    % (vec2string(refined_axis), order, fsc),
                )
                if fsc > fsc_thresh_for_trueorder:
                    i = 0
                    while fsc > fsc_thresh_for_trueorder:
                        if i > 5:
                            break
                        exponent += 1
                        i += 1
                        # 2. check for higher powers
                        previous_odr1 = current_odr1
                        current_odr1 = order**exponent
                        previous_fsc = fsc
                        if current_odr1 > 50:
                            emdalogger.log_string(
                                fobj, "Current order is greater than 50. Stop."
                            )
                            break
                        porder = previous_odr1
                        corder = current_odr1
                        angledif_list = [
                            abs(
                                float(360 / porder / 2)
                                - float(360 * j / corder)
                            )
                            for j in range(1, corder)
                        ]
                        ang = float(
                            360
                            * (angledif_list.index(min(angledif_list)) + 1)
                            / current_odr1
                        )
                        fsc_results = calc_fsc(
                            emmap1=emmap1,
                            axis=refined_axis,
                            angle=ang,
                            t=t,
                        )
                        if fsc_results is not None:
                            fsc = fsc_results[1]
                            emmap1.symdat.append(
                                [
                                    [corder],
                                    refined_axis,
                                    list(fsc_results[0]),
                                    list(t),
                                ]
                            )
                        else:
                            emdalogger.log_string(
                                fobj,
                                (
                                    "Something went wrong in FSC calculation."
                                    " Stop"
                                ),
                            )
                            break
                        emdalogger.log_string(
                            fobj,
                            "   Axis: %s   Order: %s   FSC: % .3f"
                            % (vec2string(refined_axis), current_odr1, fsc),
                        )
                    true_order_list.append(previous_odr1)
                    true_fsc_list.append(previous_fsc)
                else:
                    print("Avg FSC= ", fsc)
            else:
                emdalogger.log_string(
                    fobj,
                    (
                        "Something went wrong with FSC calculation."
                        "continue with next order"
                    ),
                )
                continue
        bestorder, bestorder_fsc = check_maxorder(
            [emmap1, true_order_list, true_fsc_list, refined_axis, t, fobj]
        )
        return bestorder, bestorder_fsc
    except:
        fobj.write(traceback.format_exc())
        raise ValueError


def check_maxorder(params):
    (emmap1, true_order_list, true_fsc_list, refined_axis, t, fobj) = params
    try:
        if len(true_order_list) == 0:
            bestorder = 1
            bestorder_fsc = 0.0
        elif len(true_order_list) == 1:
            bestorder = true_order_list[0]
            bestorder_fsc = true_fsc_list[0]
            emdalogger.log_string(
                fobj,
                "   Axis: %s   Best order: %s   FSC: % .3f"
                % (vec2string(refined_axis), bestorder, bestorder_fsc),
            )
        elif len(true_order_list) == 2:
            maxorder = max(true_order_list)
            corder = current_order = np.prod(true_order_list)
            angledif_list = [
                abs(float(360 / maxorder / 2) - float(360 * j / corder))
                for j in range(1, corder)
            ]
            ang = float(
                360
                * (angledif_list.index(min(angledif_list)) + 1)
                / current_order
            )
            fsc_results = calc_fsc(
                emmap1=emmap1, axis=refined_axis, angle=ang, t=t
            )
            fsc = fsc_results[1]
            emmap1.symdat.append(
                [[corder], refined_axis, list(fsc_results[0]), list(t)]
            )
            emdalogger.log_string(
                fobj,
                "   Axis: %s   Order: %s   FSC: % .3f"
                % (vec2string(refined_axis), current_order, fsc),
            )
            if fsc < pg_decide_fsc:
                bestorder = true_order_list[np.argmax(true_fsc_list)]
                bestorder_fsc = max(true_fsc_list)
            else:
                bestorder = current_order
                bestorder_fsc = fsc
        else:
            # there are more than 2 orders
            # need to check with a group compatibility table
            new_order_list = []
            new_fsc_list = []
            maxorder = max(true_order_list)
            for r in range(2, len(true_order_list)):
                l = itertools.combinations(true_order_list, r)
                for itm in l:
                    new_order_list.append(np.prod(itm))
            for odr in new_order_list:
                angledif_list = [
                    abs(float(360 / maxorder / 2) - float(360 * j / odr))
                    for j in range(1, odr)
                ]
                ang = float(
                    360 * (angledif_list.index(min(angledif_list)) + 1) / odr
                )
                fsc_results = calc_fsc(
                    emmap1=emmap1,
                    axis=refined_axis,
                    angle=ang,
                    t=t,
                )
                fsc = fsc_results[1]
                emmap1.symdat.append(
                    [[odr], refined_axis, list(fsc_results[0]), list(t)]
                )
                emdalogger.log_string(
                    fobj,
                    "   Axis: %s   Order: %s   FSC: % .3f"
                    % (vec2string(refined_axis), odr, fsc),
                )
                new_fsc_list.append(fsc)
                emdalogger.log_string(
                    fobj,
                    "   Axis: %s   Order: %s   FSC: % .3f"
                    % (vec2string(refined_axis), odr, fsc),
                )
            possible_orders = []
            possible_fscs = []
            for i, ifsc in enumerate(new_fsc_list):
                if ifsc > min(true_fsc_list):
                    possible_orders.append(new_order_list[i])
                    possible_fscs.append(ifsc)
            bestorder = max(possible_orders)
            bestorder_fsc = possible_fscs[np.argmax(possible_orders)]
            emdalogger.log_string(
                fobj,
                "   Axis: %s   Best order: %s   FSC: % .3f\n"
                % (vec2string(refined_axis), bestorder, bestorder_fsc),
            )
        return bestorder, bestorder_fsc
    except Exception as e:
        raise e


def translation_refinement(emmap1, axis, order, fobj, t=None):
    if t is None:
        t = np.array([0.0, 0.0, 0.0], "float")
    try:
        """# normalise fs
        nx, ny, nz = emmap1.claimedres_data[0].shape
        eo = fcodes2.get_normalized_sf_singlemap(
            fo=emmap1.claimedres_data[0],
            bin_idx=emmap1.claimedres_data[1],
            nbin=emmap1.claimedres_data[3],
            mode=0,
            nx=nx,ny=ny,nz=nz,
            )
        #eo = emmap1.claimedres_data[0]
        # refine translation again - sometimes there are problems in t
        print('******* Translation only refinement *******')
        f_transformed = maptools.transform_f(
                        flist=[eo],#[emmap1.claimedres_data[0]],
                        axis=axis,
                        translation=t,
                        angle=float(360/order)
                        )
        emmap2 = axis_refinement.EmmapOverlay(arr=emmap1.arr)
        emmap2.map_unit_cell = emmap1.map_unit_cell
        emmap2.bin_idx = emmap1.claimedres_data[1]
        emmap2.res_arr = emmap1.claimedres_data[4]
        emmap2.nbin = emmap1.claimedres_data[3]
        emmap2.fo_lst = [emmap1.claimedres_data[0], f_transformed[0]]
        #emmap2.eo_lst = [emmap1.claimedres_data[0], f_transformed[0]]
        emmap2.eo_lst = [eo, f_transformed[0]]
        emmap2.comlist = []
        emmap2.map_origin = [0,0,0]
        emmap2.map_dim = f_transformed[0].shape
        emmap2.pixsize = emmap1.pix
        from emda2.ext.overlay import run_fit
        results = run_fit(
            emmap1=emmap2,
            rotmat=np.identity(3),
            t=np.array([0., 0., 0.], 'float'),
            ncycles=10,
            ifit=1,
            fitres=emmap1.claimedres_data[2],
            t_only=True,
            )
        if results is not None:
            t2, q_final = results
            refined_t = t + t2
        else:
            refined_t = t
        print('t_final (after): ', refined_t)"""
        refined_t = t
        return refined_t
    except:
        fobj.write(traceback.format_exc())


def refine_ax(emmap1, axlist, orderlist, fobj):
    try:
        ref_axlist = []  # refined axes
        ref_fsclist = []  # fscs @ claimed resol. of refined axes
        ref_tlist = []  # refined translations
        ref_axposlist = []  # refined axes positions
        claimed_res = emmap1.claimed_res
        cbin = emmap1.claimed_bin  # bin corresponding to claimed resol.
        initial_t = [0.0, 0.0, 0.0]
        for i, axis in enumerate(axlist):
            fsclist = []  # FSC up to Nyquist (nbin)
            order = orderlist[i]
            binfsc, axfsc, frt = calc_fsc(
                emmap1=emmap1,
                axis=axis,
                angle=float(360 / order),
                t=initial_t,
            )
            fsclist.append(binfsc)
            if axfsc > fsc_thresh_for_axrefine:
                emdalogger.log_newline(fobj)
                emdalogger.log_string(fobj, "   Axis refinement")
                emdalogger.log_string(fobj, "---------------------")
                emdalogger.log_string(
                    fobj,
                    "   Initial axis: %s   Order: %s   FSC: % .3f @ % .2f A\n"
                    % (vec2string(axis), order, axfsc, claimed_res),
                )
                ax_fnl = axis
                t_fnl = [0.0, 0.0, 0.0]
                ax_position = []
                results = [ax_fnl, t_fnl, ax_position, fsclist[0]]
            else:
                emdalogger.log_newline(fobj)
                emdalogger.log_string(fobj, "   Axis refinement")
                emdalogger.log_string(fobj, "---------------------")
                emdalogger.log_vector(
                    fobj,
                    {
                        "Initial axis": axis,
                        "Initial trans": initial_t,
                        "Order": order,
                        "Claimed resol.": claimed_res,
                    },
                )
                emdalogger.log_string(
                    fobj, "FSC: % .3f @ % .2f (A)" % (axfsc, claimed_res)
                )
                results = axis_refinement.axis_refine(
                    emmap1=emmap1,
                    rotaxis=axis,
                    symorder=order,
                    fobj=fobj,
                    t_init=initial_t,
                    frt=frt,
                )
            ax_fnl, t_fnl, ax_position, binfsc_refax = results
            ref_axlist.append(ax_fnl)

            # determine the t_centroid for refined axis and current order
            result = [
                abs(t_fnl[i] * emmap1.map_dim[i] * emmap1.pix[i]) < 0.01
                for i in range(3)
            ]
            if all(result):
                t_centroid = t_fnl
            elif order == 2:
                t_centroid = [elem / 2 for elem in t_fnl]
            else:
                t_centroid = get_t_to_centroid(
                    emmap1=emmap1, axis=ax_fnl, order=order
                )
            # collect results in symdat
            binfsc_refax, _, _ = calc_fsc(
                emmap1=emmap1,
                axis=ax_fnl,
                angle=float(360 / order),
                t=t_centroid,
            )
            fsc_refined_ax1 = binfsc_refax[
                cbin
            ]  # refined FSC @ claimed resol.
            emmap1.symdat.append(
                [[order], ax_fnl, list(binfsc_refax), t_centroid]
            )
            pos_ax = [
                (emmap1.com1[i] - t_centroid[i] * emmap1.map_dim[i])
                * emmap1.pix[i]
                for i in range(3)
            ]
            ref_axposlist.append(pos_ax)
            ref_tlist.append(t_centroid)
            fsclist.append(binfsc_refax)
            ref_fsclist.append(fsc_refined_ax1)
            emdalogger.log_string(
                fobj,
                "Refined axis: %s   Order: %s   FSC: % .3f @ % .2f A"
                % (vec2string(ax_fnl), order, fsc_refined_ax1, claimed_res),
            )
            emdalogger.log_string(
                fobj,
                "Position of refined axis [x, y, z] (A): %s"
                % vec2string(ax_position),
            )
            emdalogger.log_string(
                fobj, "FSCs before and after axis refinement"
            )
            emdalogger.log_fsc(
                fobj,
                {
                    "Resol.": emmap1.res_arr,
                    "FSC(orig.ax)": fsclist[0],
                    "FSC(refi.ax)": fsclist[1],
                    "FSC(full)": emmap1.fscfull,
                },
            )
            emdalogger.log_newline(fobj)
        return [ref_axlist, ref_tlist, ref_axposlist, orderlist, ref_fsclist]
    except:
        fobj.write(traceback.format_exc())


def check_tetrahedral(emmap1, tetrahedral_2axes, tetrahedral_2axes_fsc, fobj):
    if len(tetrahedral_2axes) > 0:
        # there are 2-folds conform to T
        # refine the 2-fold with highest FSC
        th_axis = tetrahedral_2axes[np.argmax(tetrahedral_2axes_fsc)]
        refinement_results = refine_ax(
            emmap1=emmap1,
            axlist=[th_axis],
            orderlist=[2],
            fobj=fobj,
        )
        if refinement_results is not None:
            (ref_axlist, ref_tlist, _, _, ref_fsclist) = refinement_results
            refined_th_axis = ref_axlist[0]
            fsc_thax_refined = ref_fsclist[0]
            refined_t = ref_tlist[0]
            if fsc_thax_refined >= pg_decide_fsc:
                # check the order of the refined axis
                # t_centroid = [elem/2 for elem in refined_t]
                t_centroid = refined_t
                # get maxorder of the refined axis
                thax_bestorder, thax_bestorder_fsc = get_axorder(
                    emmap1=emmap1,
                    refined_axis=refined_th_axis,
                    order_list=[2],
                    fobj=fobj,
                    t=t_centroid,
                )
                if thax_bestorder == 2:
                    pg = "T"
                elif thax_bestorder == 4:
                    pg = "O"
                else:
                    pg = "Unknown"
                return [True, pg]
            else:
                # T 2-fold not true
                return [False]
    else:
        return [False]


def check_octahedral(emmap1, octahedral_2axes, octahedral_2axes_fsc, fobj):
    if len(octahedral_2axes) > 0:
        # there are 2-folds conform to O
        # refine the 2-fold with highest FSC
        oc_axis = octahedral_2axes[np.argmax(octahedral_2axes_fsc)]
        refinement_results = refine_ax(
            emmap1=emmap1,
            axlist=[oc_axis],
            orderlist=[2],
            fobj=fobj,
        )
        if refinement_results is not None:
            (ref_axlist, ref_tlist, _, _, ref_fsclist) = refinement_results
            refined_oc_axis = ref_axlist[0]
            fsc_ocax_refined = ref_fsclist[0]
            refined_t = ref_tlist[0]
            if fsc_ocax_refined >= pg_decide_fsc:
                pg = "O"
                return [True, pg]
            else:
                # O 2-fold not true
                return [False]
    else:
        return [False]


def five_folds(emmap1, axes, sordern, fscs, fobj):
    sorder5ax, sorder3ax, sorder2ax, sordernax = axes
    sorder5fsc, sorder3fsc, sorder2fsc, sordernfsc = fscs
    if len(sorder5ax) > 1:
        try:
            emdalogger.log_string(fobj, "Checking for I...")
            emdalogger.log_string(fobj, "Candidate axes:")
            emdalogger.log_string(
                fobj,
                "   Axis1: %s   Order: %s   FSC: % .3f"
                % (vec2string(sorder5ax[0]), 5, sorder5fsc[0]),
            )
            emdalogger.log_string(
                fobj,
                "   Axis2: %s   Order: %s   FSC: % .3f"
                % (vec2string(sorder5ax[1]), 5, sorder5fsc[1]),
            )
            angle = cosine_angle(sorder5ax[0], sorder5ax[1])
            emdalogger.log_string(fobj, "   Angle between them: % .3f" % angle)
            if (
                abs(angle - 63.47) <= ang_tol_p
                or abs(angle - 116.57) <= ang_tol_p
            ):
                # ax1, ax2 refinement
                refinement_results = refine_ax(
                    emmap1=emmap1,
                    axlist=[sorder5ax[0], sorder5ax[1]],
                    orderlist=[5, 5],
                    fobj=fobj,
                )
                if refinement_results is not None:
                    refined_ax1 = refinement_results[0][0]
                    refined_ax2 = refinement_results[0][1]
                    fsc_rax1 = refinement_results[4][0]
                    fsc_rax2 = refinement_results[4][1]
                    if fsc_rax1 >= pg_decide_fsc and fsc_rax2 >= pg_decide_fsc:
                        angle = cosine_angle(refined_ax1, refined_ax2)
                        if (
                            abs(angle - 63.47) <= ang_tol
                            or abs(angle - 116.57) <= ang_tol
                        ):
                            pg = "I"
                    elif (
                        fsc_rax1 >= pg_decide_fsc and fsc_rax2 < pg_decide_fsc
                    ):
                        # only the first 5-fold axis is valid
                        # possible point groups: C5 or D5
                        sorder5ax = [refined_ax1]
                        sorder5fsc = [fsc_rax1]
                        axes = [sorder5ax, sorder3ax, sorder2ax, sordernax]
                        fscs = [sorder5fsc, sorder3fsc, sorder2fsc, sordernfsc]
                        pg = single_5fold(
                            emmap1=emmap1,
                            axes=axes,
                            fscs=fscs,
                            sordern=sordern,
                            fobj=fobj,
                        )
                    elif (
                        fsc_rax1 < pg_decide_fsc and fsc_rax2 >= pg_decide_fsc
                    ):
                        # only the second 5-fold axis is valid
                        # possible point groups: C5 or D5
                        sorder5ax = [refined_ax2]
                        sorder5fsc = [fsc_rax2]
                        axes = [sorder5ax, sorder3ax, sorder2ax, sordernax]
                        fscs = [sorder5fsc, sorder3fsc, sorder2fsc, sordernfsc]
                        pg = single_5fold(
                            emmap1=emmap1,
                            axes=axes,
                            fscs=fscs,
                            sordern=sordern,
                            fobj=fobj,
                        )
                    else:
                        # none of the 5-folds are valid. check for lower symmetries
                        axes = [sorder3ax, sorder2ax, sordernax]
                        fscs = [sorder3fsc, sorder2fsc, sordernfsc]
                        pg = no5folds(
                            emmap1=emmap1,
                            axes=axes,
                            sordern=sordern,
                            fscs=fscs,
                            fobj=fobj,
                        )
                else:
                    pg = single_5fold(
                        emmap1=emmap1,
                        axes=axes,
                        fscs=fscs,
                        sordern=sordern,
                        fobj=fobj,
                    )
            else:
                pg = single_5fold(
                    emmap1=emmap1,
                    axes=axes,
                    fscs=fscs,
                    sordern=sordern,
                    fobj=fobj,
                )
        except:
            fobj.write(traceback.format_exc())
            pg = single_5fold(
                emmap1=emmap1, axes=axes, fscs=fscs, sordern=sordern, fobj=fobj
            )
    elif len(sorder5ax) == 1:
        pg = single_5fold(
            emmap1=emmap1, axes=axes, fscs=fscs, sordern=sordern, fobj=fobj
        )
    else:
        axes = [sorder3ax, sorder2ax, sordernax]
        fscs = [sorder3fsc, sorder2fsc, sordernfsc]
        pg = no5folds(
            emmap1=emmap1, axes=axes, sordern=sordern, fscs=fscs, fobj=fobj
        )
    return pg


def check_for_cyclic_only(emmap1, axes, orders, fscs, fobj):
    # to be cyclic, there should be only one axis
    # but there can be more than one primitive order
    # axes = [axis for sublist in axes for axis in sublist]
    # orders = [order for sublist in orders for order in sublist]
    # fscs = [fsc for sublist in fscs for fsc in sublist]
    sfscs, sorders, saxes = sort_together([fscs, orders, axes], reverse=True)
    # only one axis?
    mask = [True]
    if len(saxes) == 1:
        # only one axis
        pg = "C1"
        refinement_results = refine_ax(
            emmap1=emmap1,
            axlist=[saxes[0]],
            orderlist=[sorders[0]],
            fobj=fobj,
        )
        if refinement_results is not None:
            (ref_axlist, ref_tlist, _, _, ref_fsclist) = refinement_results
            refined_main_axis = ref_axlist[0]
            fsc_refined_axis = ref_fsclist[0]
            t_centroid = ref_tlist[0]
            if fsc_refined_axis > pg_decide_fsc:
                emdalogger.log_string(
                    fobj, "%s-fold is real. Checking for C..." % sorders[0]
                )
                refined_t = ref_tlist[0]
                # get the best order of main axis
                mainax_bestorder, mainax_bestorder_fsc = get_axorder(
                    emmap1=emmap1,
                    refined_axis=refined_main_axis,
                    order_list=sorders,
                    fobj=fobj,
                    t=refined_t,
                )
                pg = "C%s" % mainax_bestorder
        return pg
    # more than one axis
    for i, axis in enumerate(saxes[1:]):
        angle = cosine_angle(saxes[0], axis)
        print("Angle between Ax0 and Ax%i is %f" % (i + 1, angle))
        if angle <= ang_tol_p or (180.0 - angle) <= ang_tol_p:
            mask.append(True)
        else:
            mask.append(False)
    if all(mask):
        # only one axis, so it should be cyclic
        # refine the main axis
        refinement_results = refine_ax(
            emmap1=emmap1,
            axlist=[saxes[0]],
            orderlist=[sorders[0]],
            fobj=fobj,
        )
        if refinement_results is not None:
            (ref_axlist, ref_tlist, _, _, ref_fsclist) = refinement_results
            refined_main_axis = ref_axlist[0]
            fsc_refined_axis = ref_fsclist[0]
            t_centroid = ref_tlist[0]
            if fsc_refined_axis > pg_decide_fsc:
                emdalogger.log_string(
                    fobj, "%s-fold is real. Checking for C..." % sorders[0]
                )
                refined_t = ref_tlist[0]
                # get the best order of main axis
                mainax_bestorder, mainax_bestorder_fsc = get_axorder(
                    emmap1=emmap1,
                    refined_axis=refined_main_axis,
                    order_list=sorders,
                    fobj=fobj,
                    t=refined_t,
                )
                pg = "C%s" % mainax_bestorder
            else:
                ll = []
                for order in sorders[1:]:
                    binfsc_refax, _, _ = calc_fsc(
                        emmap1=emmap1,
                        axis=refined_main_axis,
                        angle=float(360 / order),
                        t=t_centroid,
                    )
                    fsc_refined_ax1 = binfsc_refax[
                        emmap1.claimed_bin
                    ]  # refined FSC @ claimed resol.
                    ll.append(fsc_refined_ax1)
                    emmap1.symdat.append(
                        [
                            [order],
                            refined_main_axis,
                            list(binfsc_refax),
                            t_centroid,
                        ]
                    )
                reduced_list = [
                    sorders[i + 1]
                    for i, ifsc in enumerate(ll)
                    if ifsc > pg_decide_fsc
                ]
                if len(reduced_list) > 0:
                    mainax_bestorder, _ = get_axorder(
                        emmap1=emmap1,
                        refined_axis=refined_main_axis,
                        order_list=reduced_list,
                        fobj=fobj,
                        t=t_centroid,
                    )
                    pg = "C%s" % mainax_bestorder
                else:
                    pg = "C1"
        else:
            pg = None
    else:
        pg = None
    return pg


def single_5fold(emmap1, axes, sordern, fscs, fobj):
    try:
        emdalogger.log_string(fobj, "Single 5-fold found")
        sorder5ax, sorder3ax, sorder2ax, sordernax = axes
        sorder5fsc, sorder3fsc, sorder2fsc, sordernfsc = fscs
        # check if the 5-fold is real
        # refine the 5-fold axis
        refinement_results = refine_ax(
            emmap1=emmap1,
            axlist=[sorder5ax[0]],
            orderlist=[5],
            fobj=fobj,
        )
        if refinement_results is not None:
            (ref_axlist, ref_tlist, _, _, ref_fsclist) = refinement_results
            fsc_refined_5fold = ref_fsclist[0]
            if fsc_refined_5fold > pg_decide_fsc:
                emdalogger.log_string(
                    fobj, "5-fold is real. Checking for D or C..."
                )
                refined_main_axis = ref_axlist[0]  # sorder5ax[0]
                refined_mainax_fsc = ref_fsclist[0]  # sorder5fsc[0]
                refined_t = ref_tlist[0]
                parellel_orders = [5]
                parellel_fscs = [sorder5fsc[0]]
                perpendicular_orders = []
                perpendicular_axes = []
                perpendicular_fscs = []
                # fobj.write('   Candidate axes:\n')
                emdalogger.log_string(fobj, "   Candidate axes:")
                if len(sorder3ax) > 0:
                    for i, ax in enumerate(sorder3ax):
                        emdalogger.log_string(
                            fobj,
                            "   Axis1: %s   Order: %s   FSC: % .3f"
                            % (
                                vec2string(refined_main_axis),
                                5,
                                refined_mainax_fsc,
                            ),
                        )
                        emdalogger.log_string(
                            fobj,
                            "   Axis2: %s   Order: %s   FSC: % .3f"
                            % (vec2string(ax), 3, sorder3fsc[i]),
                        )

                        angle = cosine_angle(refined_main_axis, ax)
                        emdalogger.log_string(
                            fobj, "   Angle between them: % .3f" % angle
                        )
                        if angle <= ang_tol_p or (180.0 - angle) <= ang_tol_p:
                            parellel_orders.append(3)
                            parellel_fscs.append(sorder3fsc[i])
                        elif abs(angle - 90.0) < ang_tol:
                            perpendicular_orders.append(3)
                            perpendicular_axes.append(ax)
                            perpendicular_fscs.append(sorder3fsc[i])
                        else:
                            continue
                if len(sorder2ax) > 0:
                    for i, ax in enumerate(sorder2ax):
                        emdalogger.log_string(
                            fobj,
                            "   Axis1: %s   Order: %s   FSC: % .3f"
                            % (
                                vec2string(refined_main_axis),
                                5,
                                refined_mainax_fsc,
                            ),
                        )
                        emdalogger.log_string(
                            fobj,
                            "   Axis2: %s   Order: %s   FSC: % .3f"
                            % (vec2string(ax), 2, sorder2fsc[i]),
                        )
                        angle = cosine_angle(refined_main_axis, ax)
                        emdalogger.log_string(
                            fobj, "   Angle between them: % .3f" % angle
                        )
                        if angle < ang_tol_p or (180.0 - angle) < ang_tol_p:
                            # ax is same as 5fold
                            parellel_orders.append(2)
                            parellel_fscs.append(sorder2fsc[i])
                        elif abs(angle - 90.0) < ang_tol:
                            # ax is perpendicular to 5fold
                            perpendicular_orders.append(2)
                            perpendicular_axes.append(ax)
                            perpendicular_fscs.append(sorder2fsc[i])
                        else:
                            continue
                if len(sordernax) > 0:
                    for i, ax in enumerate(sordernax):
                        emdalogger.log_string(
                            fobj,
                            "   Axis1: %s   Order: %s   FSC: % .3f"
                            % (
                                vec2string(refined_main_axis),
                                5,
                                refined_mainax_fsc,
                            ),
                        )
                        emdalogger.log_string(
                            fobj,
                            "   Axis2: %s   Order: %s   FSC: % .3f"
                            % (vec2string(ax), sordern[i], sordernfsc[i]),
                        )
                        angle = cosine_angle(refined_main_axis, ax)
                        emdalogger.log_string(
                            fobj, "   Angle between them: % .3f" % angle
                        )
                        if angle < ang_tol_p or (180.0 - angle) < ang_tol_p:
                            parellel_orders.append(sordern[i])
                            parellel_fscs.append(sordernfsc[i])
                        elif abs(angle - 90.0) < ang_tol:
                            # ax is perpendicular to 5fold
                            perpendicular_orders.append(sordern[i])
                            perpendicular_axes.append(ax)
                            perpendicular_fscs.append(sordernfsc[i])
                        else:
                            continue
                if len(parellel_orders) > 1:
                    emdalogger.log_string(
                        fobj, "parellel_orders: %s" % parellel_orders
                    )
                    """ result = [
                        abs(refined_t[i]*emmap1.map_dim[i]*emmap1.pix[i]) < 0.05 for i in range(3)]
                    if all(result):
                        t_centroid = refined_t
                    else:
                        # refine the translation for all copies of the 5-fold
                        # then use the centroid of the polygon for translation
                        t_centroid = get_t_to_centroid(
                            emmap1=emmap1, axis=refined_main_axis, order=5) """
                    t_centroid = refined_t
                    # get the order of the main axis
                    mainax_bestorder, mainax_bestorder_fsc = get_axorder(
                        emmap1=emmap1,
                        refined_axis=refined_main_axis,
                        order_list=parellel_orders,
                        fobj=fobj,
                        t=t_centroid,
                    )
                    emdalogger.log_string(
                        fobj,
                        "mainax bestorder=%i FSC=% .3f"
                        % (mainax_bestorder, mainax_bestorder_fsc),
                    )
                else:
                    # only main axis
                    # using the refined_main_axis determine the best order for that axis
                    mainax_bestorder, mainax_bestorder_fsc = get_axorder(
                        emmap1=emmap1,
                        refined_axis=refined_main_axis,
                        order_list=parellel_orders,
                        fobj=fobj,
                        t=refined_t,
                    )
                if mainax_bestorder_fsc >= pg_decide_fsc:
                    # mainax is real
                    if len(perpendicular_axes) > 0:
                        # sort by fsc
                        (
                            sorted_perpendicular_fscs,
                            sorted_perpendicular_axes,
                        ) = sort_together(
                            [perpendicular_fscs, perpendicular_axes],
                            reverse=True,
                        )
                        refinement_results = refine_ax(
                            emmap1=emmap1,
                            axlist=[sorted_perpendicular_axes[0]],
                            orderlist=[2],  # perp axes are 2-folds
                            fobj=fobj,
                        )
                        if refinement_results is not None:
                            (
                                ref_axlist,
                                ref_tlist,
                                _,
                                _,
                                ref_fsclist,
                            ) = refinement_results
                            refined_perpend_axis = ref_axlist[0]
                            fsc_perpax_refined = ref_fsclist[0]
                            if fsc_perpax_refined >= pg_decide_fsc:
                                # perp axis is real
                                pg = decide_pointgroup(
                                    axeslist=[
                                        refined_main_axis,
                                        refined_perpend_axis,
                                    ],
                                    orderlist=[mainax_bestorder, 2],
                                )[0]
                            else:
                                # perp axis is not real
                                pg = "C" + str(mainax_bestorder)
                        else:
                            pg = "C" + str(mainax_bestorder)
                    else:
                        # no perp axes, but main ax is real
                        pg = "C" + str(mainax_bestorder)
                else:
                    # main ax is not real
                    axes = [sorder3ax, sorder2ax, sordernax]
                    fscs = [sorder3fsc, sorder2fsc, sordernfsc]
                    pg = no5folds(
                        emmap1=emmap1,
                        axes=axes,
                        sordern=sordern,
                        fscs=fscs,
                        fobj=fobj,
                    )
            else:
                # 5-fold is not real
                axes = [sorder3ax, sorder2ax, sordernax]
                fscs = [sorder3fsc, sorder2fsc, sordernfsc]
                pg = no5folds(
                    emmap1=emmap1,
                    axes=axes,
                    sordern=sordern,
                    fscs=fscs,
                    fobj=fobj,
                )
        else:
            axes = [sorder3ax, sorder2ax, sordernax]
            fscs = [sorder3fsc, sorder2fsc, sordernfsc]
            pg = no5folds(
                emmap1=emmap1, axes=axes, sordern=sordern, fscs=fscs, fobj=fobj
            )
    except:
        emdalogger.log_tracebck(fobj, traceback.format_exc())
        axes = [sorder3ax, sorder2ax, sordernax]
        fscs = [sorder3fsc, sorder2fsc, sordernfsc]
        pg = no5folds(
            emmap1=emmap1, axes=axes, sordern=sordern, fscs=fscs, fobj=fobj
        )
    return pg


def no5folds(emmap1, axes, sordern, fscs, fobj):
    sorder3ax, sorder2ax, sordernax = axes
    sorder3fsc, sorder2fsc, sordernfsc = fscs
    if len(sorder3ax) > 1:
        # more than one 3-fold axes
        try:
            emdalogger.log_string(fobj, "Checking for O/T ...")
            temp_3ax_list = []
            for i, ax in enumerate(sorder3ax):
                if i > 0:
                    emdalogger.log_string(fobj, "Candidate axes:")
                    emdalogger.log_string(
                        fobj,
                        "   Axis1: %s   Order: %s   FSC: % .3f"
                        % (vec2string(sorder3ax[0]), 3, sorder3fsc[0]),
                    )
                    emdalogger.log_string(
                        fobj,
                        "   Axis2: %s   Order: %s   FSC: % .3f"
                        % (vec2string(sorder3ax[i]), 3, sorder3fsc[i]),
                    )
                    angle = cosine_angle(sorder3ax[0], sorder3ax[i])
                    fobj.write("   Angle between them: % .3f" % angle)
                    emdalogger.log_string(
                        fobj, "   Angle between them: % .3f" % angle
                    )
                    if (
                        abs(angle - 109.47) <= ang_tol_p
                        or abs(angle - 70.53) <= ang_tol_p
                    ):
                        temp_3ax_list.append(sorder3ax[i])
            if len(temp_3ax_list) > 0:
                # refine both 3-fold axes
                refinement_results = refine_ax(
                    emmap1=emmap1,
                    axlist=[sorder3ax[0], temp_3ax_list[0]],
                    orderlist=[3, 3],
                    fobj=fobj,
                )
                if refinement_results is not None:
                    (
                        ref_axlist,
                        ref_tlist,
                        _,
                        _,
                        ref_fsclist,
                    ) = refinement_results
                    refined_ax1_o3, refined_ax2_o3 = (
                        ref_axlist[0],
                        ref_axlist[1],
                    )
                    fsc_refined_ax1, fsc_refined_ax2 = (
                        ref_fsclist[0],
                        ref_fsclist[1],
                    )
                    if (
                        fsc_refined_ax1 >= pg_decide_fsc
                        and fsc_refined_ax2 >= pg_decide_fsc
                    ):
                        # both 3-fld axes are valid
                        angle = cosine_angle(refined_ax1_o3, refined_ax2_o3)
                        if (
                            abs(angle - 109.47) <= ang_tol
                            or abs(angle - 70.53) <= ang_tol
                        ):
                            # both 3-fld axes are valid for T or O
                            # distiction should be made using 4 folds
                            if len(sorder2ax) > 0:
                                # at least one 2-fold axis present
                                order4ax = []
                                order2ax_tmp = []
                                for ax in sorder2ax:  # loop 3
                                    refinement_results = refine_ax(
                                        emmap1=emmap1,
                                        axlist=[ax],
                                        orderlist=[2],
                                        fobj=fobj,
                                    )
                                    if refinement_results is not None:
                                        (
                                            ref_axlist,
                                            ref_tlist,
                                            _,
                                            _,
                                            ref_fsclist,
                                        ) = refinement_results
                                        refined_ax = ref_axlist[0]
                                        fsc_refined_ax = ref_fsclist[0]
                                        refined_t = ref_tlist[0]
                                        # get the centroid_t
                                        # t_centroid = [elem/2 for elem in refined_t]
                                        t_centroid = refined_t
                                        if fsc_refined_ax >= pg_decide_fsc:
                                            (
                                                axorder,
                                                ax_bestoder_fsc,
                                            ) = get_axorder(
                                                emmap1=emmap1,
                                                refined_axis=refined_ax,
                                                order_list=[2],
                                                t=t_centroid,
                                                fobj=fobj,
                                            )
                                            if axorder == 4:
                                                order4ax.append(refined_ax)
                                                if len(order4ax) > 1:
                                                    break  # l3
                                            else:
                                                order2ax_tmp.append(refined_ax)
                                        else:
                                            continue
                                    else:
                                        continue
                                if len(order4ax) > 1:
                                    refined_ax1_o4 = order4ax[0]
                                    refined_ax2_o4 = order4ax[1]
                                    angle = cosine_angle(
                                        refined_ax1_o4, refined_ax2_o4
                                    )
                                    assert abs(angle - 90.0) < ang_tol
                                    pg = "O"
                                    emdalogger.log_string(
                                        fobj, "Pointgroup: %s" % (pg)
                                    )
                                    emdalogger.log_string(
                                        fobj, "Group generator axes:"
                                    )
                                    emdalogger.log_string(
                                        fobj,
                                        "   Axis1: %s   Order: %s"
                                        % (vec2string(refined_ax1_o4), 4),
                                    )
                                    emdalogger.log_string(
                                        fobj,
                                        "   Axis2: %s   Order: %s"
                                        % (vec2string(refined_ax2_o4), 4),
                                    )
                                    return pg
                                else:
                                    assert len(order2ax_tmp) > 0
                                    pg = "T"
                                    angle = cosine_angle(
                                        refined_ax1_o3, refined_ax2_o3
                                    )
                                    emdalogger.log_string(
                                        fobj, "Pointgroup: %s" % (pg)
                                    )
                                    emdalogger.log_string(
                                        fobj, "Group generator axes:"
                                    )
                                    emdalogger.log_string(
                                        fobj,
                                        "   Axis1: %s   Order: %s"
                                        % (vec2string(refined_ax1_o3), 3),
                                    )
                                    emdalogger.log_string(
                                        fobj,
                                        "   Axis2: %s   Order: %s"
                                        % (vec2string(refined_ax2_o3), 3),
                                    )
                                    return pg
                            else:
                                # no 2-fold axes.
                                emdalogger.log_string(
                                    fobj, "3-folds are present."
                                )
                                emdalogger.log_string(
                                    fobj,
                                    (
                                        "They have neccessory relationship"
                                        " for O/T"
                                    ),
                                )
                                emdalogger.log_string(
                                    fobj, "Neccessory 2-folds are missing."
                                )
                                pg = "Uknown-3"
                                return pg
                        else:
                            # refined 3-folds do not conform to T or O
                            # treat as if there's one 3-fold
                            pg = single3fold(
                                emmap1=emmap1,
                                axes=axes,
                                fscs=fscs,
                                sordern=sordern,
                                fobj=fobj,
                            )
                    elif (
                        fsc_refined_ax1 >= pg_decide_fsc
                        and fsc_refined_ax2 < pg_decide_fsc
                    ):
                        # only the first 3-fold is true
                        sorder3ax = [refined_ax1_o3]
                        sorder3fsc = [fsc_refined_ax1]
                        axes = [sorder3ax, sorder2ax, sordernax]
                        fscs = [sorder3fsc, sorder2fsc, sordernfsc]
                        pg = single3fold(
                            emmap1=emmap1,
                            axes=axes,
                            fscs=fscs,
                            sordern=sordern,
                            fobj=fobj,
                        )
                    elif (
                        fsc_refined_ax1 < pg_decide_fsc
                        and fsc_refined_ax2 >= pg_decide_fsc
                    ):
                        # only the second 3-fold is true
                        sorder3ax = [refined_ax2_o3]
                        sorder3fsc = [fsc_refined_ax2]
                        axes = [sorder3ax, sorder2ax, sordernax]
                        fscs = [sorder3fsc, sorder2fsc, sordernfsc]
                        pg = single3fold(
                            emmap1=emmap1,
                            axes=axes,
                            fscs=fscs,
                            sordern=sordern,
                            fobj=fobj,
                        )
                    else:
                        # both 3-folds are not true
                        pg = no53folds(emmap1, axes, sordern, fscs, fobj)
                else:
                    pg = single3fold(
                        emmap1=emmap1,
                        axes=axes,
                        fscs=fscs,
                        sordern=sordern,
                        fobj=fobj,
                    )
            else:
                # 3-folds do not conform to T or O
                # treat as if there's one 3-fold
                pg = single3fold(
                    emmap1=emmap1,
                    axes=axes,
                    fscs=fscs,
                    sordern=sordern,
                    fobj=fobj,
                )
        except:
            fobj.write(traceback.format_exc())
            pg = single3fold(
                emmap1=emmap1, axes=axes, fscs=fscs, sordern=sordern, fobj=fobj
            )
    elif len(sorder3ax) == 1:
        # only one 3-fold axis
        pg = single3fold(
            emmap1=emmap1, axes=axes, fscs=fscs, sordern=sordern, fobj=fobj
        )
    else:
        # no 3-fold axes
        axes = [sorder2ax, sordernax]
        fscs = [sorder2fsc, sordernfsc]
        pg = no53folds(emmap1, axes, sordern, fscs, fobj)
    return pg


def single3fold(emmap1, axes, fscs, sordern, fobj):
    sorder3ax, sorder2ax, sordernax = axes
    sorder3fsc, sorder2fsc, sordernfsc = fscs
    # only one 3-fold axis
    try:
        # check if the 3-fold is true
        # refine main axis
        refinement_results = refine_ax(
            emmap1=emmap1,
            axlist=[sorder3ax[0]],
            orderlist=[3],
            fobj=fobj,
        )
        if refinement_results is not None:
            (ref_axlist, ref_tlist, _, _, ref_fsclist) = refinement_results
            refined_main_axis = ref_axlist[0]
            fsc_mainax_refined = ref_fsclist[0]
            refined_t = ref_tlist[0]
            if fsc_mainax_refined > pg_decide_fsc:
                emdalogger.log_string(fobj, "Checking for D/C ...")
                main_axis = refined_main_axis  # sorder3ax[0]
                parellel_orders = [3]
                parellel_fscs = [fsc_mainax_refined]  # [sorder3fsc[0]]
                perpendicular_orders = []
                perpendicular_axes = []
                perpendicular_fscs = []
                tetrahedral_2axes = []
                tetrahedral_2axes_fsc = []
                octahedral_2axes = []
                octahedral_2axes_fsc = []
                emdalogger.log_string(fobj, "Candidate axes:")
                if len(sorder2ax) > 0:
                    for i, ax in enumerate(sorder2ax):
                        emdalogger.log_string(
                            fobj,
                            "   Axis1: %s   Order: %s   FSC: % .3f"
                            % (vec2string(main_axis), 3, fsc_mainax_refined),
                        )
                        emdalogger.log_string(
                            fobj,
                            "   Axis2: %s   Order: %s   FSC: % .3f"
                            % (vec2string(ax), 2, sorder2fsc[i]),
                        )
                        angle = cosine_angle(ax, main_axis)
                        emdalogger.log_string(
                            fobj, "   Angle between them: % .3f" % angle
                        )

                        # NOTE- check 2 and 3 conform to T. e.g. EMD-21185
                        # In this case, proshade output one 3-fold and one 2-fold
                        if (
                            abs(angle - 54.74) <= ang_tol_p
                            or abs(angle - 125.26) <= ang_tol_p
                        ):
                            # tetrahedral angle
                            tetrahedral_2axes.append(ax)
                            tetrahedral_2axes_fsc.append(sorder2fsc[i])
                        elif (
                            abs(angle - 35.26) <= ang_tol_p
                            or abs(angle - 144.74) <= ang_tol_p
                        ):
                            # octahedral angle
                            octahedral_2axes.append(ax)
                            octahedral_2axes_fsc.append(sorder2fsc[i])
                        elif (
                            abs(angle) < ang_tol_p
                            or abs(180.0 - angle) < ang_tol_p
                        ):
                            # ax is same as 3fold
                            parellel_orders.append(2)
                            parellel_fscs.append(sorder2fsc[i])
                        elif abs(angle - 90.0) < ang_tol:
                            # ax is perpendicular to 3fold
                            perpendicular_orders.append(2)
                            perpendicular_axes.append(ax)
                            perpendicular_fscs.append(sorder2fsc[i])
                        else:
                            continue

                if len(sordernax) > 0:
                    for i, ax in enumerate(sordernax):
                        emdalogger.log_string(
                            fobj,
                            "   Axis1: %s   Order: %s   FSC: % .3f"
                            % (vec2string(main_axis), 3, fsc_mainax_refined),
                        )
                        emdalogger.log_string(
                            fobj,
                            "   Axis2: %s   Order: %s   FSC: % .3f"
                            % (vec2string(ax), sordern[i], sordernfsc[i]),
                        )

                        angle = cosine_angle(main_axis, ax)
                        emdalogger.log_string(
                            fobj, "   Angle between them: % .3f" % angle
                        )
                        if angle < ang_tol_p or (180.0 - angle) < ang_tol_p:
                            parellel_orders.append(sordern[i])
                            parellel_fscs.append(sordernfsc[i])
                        elif abs(angle - 90.0) < ang_tol:
                            perpendicular_orders.append(sordern[i])
                            perpendicular_axes.append(ax)
                            perpendicular_fscs.append(sordernfsc[i])
                        else:
                            continue
                emdalogger.log_newline(fobj)

                if len(octahedral_2axes) > 0:
                    results = check_octahedral(
                        emmap1, octahedral_2axes, octahedral_2axes_fsc, fobj
                    )
                    if results[0]:
                        pg = results[1]
                        return pg
                if len(tetrahedral_2axes) > 0:
                    results = check_tetrahedral(
                        emmap1, tetrahedral_2axes, tetrahedral_2axes_fsc, fobj
                    )
                    if results[0]:
                        pg = results[1]
                        return pg

                """ result = [
                    abs(refined_t[i]*emmap1.map_dim[i]*emmap1.pix[i]) < 0.05 for i in range(3)]
                if all(result):
                    t_centroid = refined_t
                else:
                    # refine the translation for all copies of the 3-fold
                    # then use the centroid of the polygon for translation
                    t_centroid = get_t_to_centroid(
                        emmap1=emmap1, axis=refined_main_axis, order=3) """
                t_centroid = refined_t
                # get the main_axis order
                mainax_bestorder, mainax_bestorder_fsc = get_axorder(
                    emmap1=emmap1,
                    refined_axis=refined_main_axis,
                    order_list=parellel_orders,
                    fobj=fobj,
                    t=t_centroid,
                )

                # pg without perpendicular axes
                pg = decide_pointgroup(
                    axeslist=[refined_main_axis], orderlist=[mainax_bestorder]
                )[0]

                # perpendicular axis
                if len(perpendicular_axes) > 0:
                    perp_ax_order = perpendicular_orders[
                        np.argmax(perpendicular_fscs)
                    ]
                    perp_axis = perpendicular_axes[
                        np.argmax(perpendicular_fscs)
                    ]
                    # refine perpendicular axis
                    refinement_results = refine_ax(
                        emmap1=emmap1,
                        axlist=[perp_axis],
                        orderlist=[perp_ax_order],
                        fobj=fobj,
                    )
                    if refinement_results is not None:
                        (
                            ref_axlist,
                            ref_tlist,
                            _,
                            _,
                            ref_fsclist,
                        ) = refinement_results
                        refined_perpend_axis = ref_axlist[0]
                        fsc_perpax_refined = ref_fsclist[0]
                        refined_t = ref_tlist[0]
                        if fsc_perpax_refined >= pg_decide_fsc:
                            # perpendicular axis is true
                            """t_centroid = [elem/2 for elem in refined_t] # true if perp_ax_order=2
                            # get the max order of the perp axis
                            perp_bestorder, perp_bestorder_fsc = get_axorder(
                                emmap1=emmap1,
                                refined_axis=refined_perpend_axis,
                                order_list=perpendicular_orders,
                                fobj=fobj,
                                t=t_centroid,
                            )
                            pg = decide_pointgroup(
                                axeslist=[refined_main_axis,
                                          refined_perpend_axis],
                                orderlist=[mainax_bestorder, perp_bestorder]
                            )[0]"""
                            # since the main_ax order is > 2, perpendicular axis order should be 2
                            # for dihedral symmetries.
                            pg = decide_pointgroup(
                                axeslist=[
                                    refined_main_axis,
                                    refined_perpend_axis,
                                ],
                                orderlist=[mainax_bestorder, 2],
                            )[0]
            else:
                # 3-fold is not real
                axes = [sorder2ax, sordernax]
                fscs = [sorder2fsc, sordernfsc]
                pg = no53folds(emmap1, axes, sordern, fscs, fobj)
        else:
            axes = [sorder2ax, sordernax]
            fscs = [sorder2fsc, sordernfsc]
            pg = no53folds(emmap1, axes, sordern, fscs, fobj)
        return pg
    except:
        fobj.write(traceback.format_exc())
        axes = [sorder2ax, sordernax]
        fscs = [sorder2fsc, sordernfsc]
        pg = no53folds(emmap1, axes, sordern, fscs, fobj)
        return pg


def no53folds(emmap1, axes, sordern, fscs, fobj):
    sorder2ax, sordernax = axes
    sorder2fsc, sordernfsc = fscs
    if len(sordernax) > 0:
        if len(sorder2ax) > 0:
            # 2-folds are there
            pg = branch1(emmap1, axes, sordern, fscs, fobj)
        else:
            # just n-folds
            pg = justnfolds(emmap1, sordernax, sordernfsc, sordern, fobj)
    else:
        # no n-folds
        if len(sorder2ax) > 0:
            # just 2-folds
            pg = just2folds(emmap1, sorder2ax, sorder2fsc, fobj)
        else:
            # neither 2 nor n-folds
            pg = "C1"
    return pg


def branch1(emmap1, axes, sordern, fscs, fobj):
    sorder2ax, sordernax = axes
    sorder2fsc, sordernfsc = fscs
    try:
        mainax = sordernax[0]
        mainax_order = sordern[0]
        mainax_fsc = sordernfsc[0]
        # refine mainax
        refinement_results = refine_ax(
            emmap1=emmap1,
            axlist=[mainax],
            orderlist=[mainax_order],
            fobj=fobj,
        )
        if refinement_results is not None:
            (ref_axlist, ref_tlist, _, _, ref_fsclist) = refinement_results
            refined_main_axis = ref_axlist[0]
            fsc_ax1_refined = ref_fsclist[0]
            refined_t = ref_tlist[0]
            if fsc_ax1_refined >= pg_decide_fsc:
                # mainax is true
                parellel_axes_order = [mainax_order]
                parellel_axes_fscs = [fsc_ax1_refined]
                if len(sordernax) > 1:
                    emdalogger.log_string(fobj, "Candidate axes:")
                    for j, ax2 in enumerate(sordernax):
                        if j > 0:
                            emdalogger.log_string(
                                fobj,
                                "   Axis1: %s   Order: %s   FSC: % .3f"
                                % (
                                    vec2string(refined_main_axis),
                                    mainax_order,
                                    fsc_ax1_refined,
                                ),
                            )
                            emdalogger.log_string(
                                fobj,
                                "   Axis2: %s   Order: %s   FSC: % .3f"
                                % (vec2string(ax2), sordern[j], sordernfsc[j]),
                            )
                            angle = cosine_angle(refined_main_axis, ax2)
                            emdalogger.log_string(
                                fobj, "   Angle between them: % .3f" % angle
                            )
                            if (
                                abs(angle) < ang_tol_p
                                or abs(angle - 180.0) < ang_tol_p
                            ):
                                parellel_axes_order.append(sordern[j])
                                parellel_axes_fscs.append(sordernfsc[j])
                if len(sorder2ax) > 0:
                    parellel_2fold_axes = []
                    perpendicular_2fold_axes = []
                    perpendicular_2fold_fscs = []
                    emdalogger.log_string(fobj, "Candidate axes:")
                    for j, ax in enumerate(sorder2ax):
                        emdalogger.log_string(
                            fobj,
                            "   Axis1: %s   Order: %s   FSC: % .3f"
                            % (
                                vec2string(refined_main_axis),
                                mainax_order,
                                mainax_fsc,
                            ),
                        )
                        emdalogger.log_string(
                            fobj,
                            "   Axis2: %s   Order: %s   FSC: % .3f\n"
                            % (vec2string(ax), 2, sorder2fsc[j]),
                        )
                        angle = cosine_angle(refined_main_axis, ax)
                        emdalogger.log_string(
                            fobj, "   Angle between them: % .3f" % angle
                        )
                        # check for parellel or perpendicular
                        if (
                            abs(angle) < ang_tol_p
                            or abs(angle - 180.0) < ang_tol_p
                        ):
                            parellel_2fold_axes.append(ax)
                        if abs(angle - 90.0) < ang_tol_p:
                            perpendicular_2fold_axes.append(ax)
                            perpendicular_2fold_fscs.append(sorder2fsc[j])
                if len(parellel_2fold_axes) > 0:
                    parellel_axes_order.append(2)
                # print('parellel_axes_order:', parellel_axes_order)
                emdalogger.log_string(
                    fobj, "parellel_axes_order: %s" % parellel_axes_order
                )

                """ result = [
                    abs(refined_t[i]*emmap1.map_dim[i]*emmap1.pix[i]) < 0.05 for i in range(3)]
                if all(result):
                    t_centroid = refined_t
                else:
                    # get the centroid t
                    t_centroid = get_t_to_centroid(
                        emmap1=emmap1, axis=refined_main_axis, order=mainax_order) """

                t_centroid = refined_t

                # find the best order of mainax
                mainax_bestorder, mainax_bestoder_fsc = get_axorder(
                    emmap1=emmap1,
                    refined_axis=refined_main_axis,
                    order_list=parellel_axes_order,
                    fobj=fobj,
                    t=t_centroid,
                )
                if len(perpendicular_2fold_axes) > 0:
                    # there is at least one perpendicular 2-fold
                    perp_2fold_ax = perpendicular_2fold_axes[
                        np.argmax(perpendicular_2fold_fscs)
                    ]
                    refinement_results = refine_ax(
                        emmap1=emmap1,
                        axlist=[perp_2fold_ax],
                        orderlist=[2],
                        fobj=fobj,
                    )
                    if refinement_results is not None:
                        (
                            ref_axlist,
                            ref_tlist,
                            _,
                            _,
                            ref_fsclist,
                        ) = refinement_results
                        refined_ax2 = ref_axlist[0]
                        fsc_ax2_refined = ref_fsclist[0]
                        if fsc_ax2_refined >= pg_decide_fsc:
                            # perp 2-fold is valid. Decide pointgroup
                            pg = decide_pointgroup(
                                axeslist=[refined_main_axis, refined_ax2],
                                orderlist=[mainax_bestorder, 2],
                            )[0]
                        else:
                            # perp 2-fold is not valid
                            pg = "C" + str(mainax_bestorder)
                    else:
                        pg = "C" + str(mainax_bestorder)
                else:
                    # no perp 2-fold
                    pg = "C" + str(mainax_bestorder)
            else:
                # mainax is not true. Use 2-folds
                pg = just2folds(emmap1, sorder2ax, sorder2fsc, fobj)
        else:
            pg = just2folds(emmap1, sorder2ax, sorder2fsc, fobj)
        return pg
    except:
        # fobj.write(traceback.format_exc())
        emdalogger.log_tracebck(fobj, traceback.format_exc())
        pg = "Unknown-5"
        return pg


def just2folds(emmap1, sorder2ax, sorder2fsc, fobj):
    pg = None
    mainax = None
    try:
        if len(sorder2ax) > 0:
            # we cannot assume highest fsc axis is the main axis
            for i, ax in enumerate(sorder2ax):
                refinement_results = refine_ax(
                    emmap1=emmap1,
                    axlist=[ax],
                    orderlist=[2],
                    fobj=fobj,
                )
                if refinement_results is not None:
                    (
                        ref_axlist,
                        ref_tlist,
                        _,
                        _,
                        ref_fsclist,
                    ) = refinement_results
                    refined_mainax = ref_axlist[0]
                    refined_mainax_fsc = ref_fsclist[0]
                    refined_t = ref_tlist[0]
                    if refined_mainax_fsc >= pg_decide_fsc:
                        mainax = refined_mainax
                        mainax_number = i
                        break

            if mainax is None:
                pg = "C1"
                return pg

            """ t_centroid = [elem/2 for elem in refined_t]
            print('t_centroid: ', t_centroid)
            #t_centroid = get_t_to_centroid(
            #            emmap1=emmap1, axis=mainax, order=2)
            #print('t_centroid: ', t_centroid) """
            t_centroid = refined_t
            # get the best order of mainax
            mainax_bestorder, mainax_bestoder_fsc = get_axorder(
                emmap1=emmap1,
                refined_axis=refined_mainax,
                order_list=[2],
                fobj=fobj,
                t=t_centroid,
            )

            perp_2ax = []
            perp_2ax_fsc = []
            for i, ax in enumerate(sorder2ax):
                if i == mainax_number:
                    continue
                emdalogger.log_string(
                    fobj,
                    "   Axis1: %s   Order: %s   FSC: % .3f"
                    % (
                        vec2string(refined_mainax),
                        mainax_bestorder,
                        mainax_bestoder_fsc,
                    ),
                )
                emdalogger.log_string(
                    fobj,
                    "   Axis2: %s   Order: %s   FSC: % .3f\n"
                    % (vec2string(ax), 2, sorder2fsc[i]),
                )
                angle = cosine_angle(refined_mainax, ax)
                emdalogger.log_string(
                    fobj, "   Angle between them: % .3f" % angle
                )
                # check for perpendicular axes
                if abs(angle - 90.0) < ang_tol_p:
                    perp_2ax.append(ax)
                    perp_2ax_fsc.append(sorder2fsc[i])

            if len(perp_2ax) > 0:
                # there are potential perpendicular 2-folds
                if mainax_bestorder == 2:
                    perpax_bestorder_list = []
                    perpax_refined_list = []
                    for pax in perp_2ax:
                        # refine each ax
                        refinement_results = refine_ax(
                            emmap1=emmap1,
                            axlist=[pax],
                            orderlist=[2],
                            fobj=fobj,
                        )
                        if refinement_results is not None:
                            (
                                ref_axlist,
                                ref_tlist,
                                _,
                                _,
                                ref_fsclist,
                            ) = refinement_results
                            refined_perp_2ax = ref_axlist[0]
                            refined_perp_2ax_fsc = ref_fsclist[0]
                            refined_t = ref_tlist[0]
                            if refined_perp_2ax_fsc >= pg_decide_fsc:
                                perpax_refined_list.append(refined_perp_2ax)
                                # t_centroid = [elem/2 for elem in refined_t]
                                t_centroid = refined_t
                                # get the best order of perpax
                                (
                                    perp2ax_bestorder,
                                    perp2ax_bestoder_fsc,
                                ) = get_axorder(
                                    emmap1=emmap1,
                                    refined_axis=refined_perp_2ax,
                                    order_list=[2],
                                    fobj=fobj,
                                    t=t_centroid,
                                )
                                perpax_bestorder_list.append(perp2ax_bestorder)
                            else:
                                continue
                        else:
                            continue

                    if len(perpax_bestorder_list) > 0:
                        perp2ax_bestorder = max(perpax_bestorder_list)
                        perpax_refined = perpax_refined_list[
                            np.argmax(perpax_bestorder_list)
                        ]
                        # choose pointgroup
                        pg = decide_pointgroup(
                            axeslist=[refined_mainax, perpax_refined],
                            orderlist=[mainax_bestorder, perp2ax_bestorder],
                        )[0]
                    else:
                        # no perp axis true
                        pg = decide_pointgroup(
                            axeslist=[refined_mainax],
                            orderlist=[mainax_bestorder],
                        )[0]
                else:
                    assert mainax_bestorder > 2
                    # refine perpendicular axes
                    for pax in perp_2ax:
                        refinement_results = refine_ax(
                            emmap1=emmap1,
                            axlist=[pax],
                            orderlist=[2],
                            fobj=fobj,
                        )
                        if refinement_results is not None:
                            (
                                ref_axlist,
                                ref_tlist,
                                _,
                                _,
                                ref_fsclist,
                            ) = refinement_results
                            refined_perp_2ax = ref_axlist[0]
                            refined_perp_2ax_fsc = ref_fsclist[0]
                            refined_t = ref_tlist[0]
                            if refined_perp_2ax_fsc >= pg_decide_fsc:
                                # get the best order of perpax
                                """perp2ax_bestorder, perp2ax_bestoder_fsc = get_axorder(
                                    emmap1=emmap1,
                                    refined_axis=refined_perp_2ax,
                                    order_list=[2],
                                    fobj=fobj,
                                    t=refined_t,
                                )"""
                                perp2ax_bestorder = (
                                    2  # because mainax_order > 2
                                )
                                # choose pointgroup
                                pg = decide_pointgroup(
                                    axeslist=[
                                        refined_mainax,
                                        refined_perp_2ax,
                                    ],
                                    orderlist=[
                                        mainax_bestorder,
                                        perp2ax_bestorder,
                                    ],
                                )[0]
                                break
                            else:
                                continue
                        else:
                            # continue with other axis
                            continue
                    if pg is None:
                        # none of the perp axes are true
                        pg = decide_pointgroup(
                            axeslist=[refined_mainax],
                            orderlist=[mainax_bestorder],
                        )[0]
            else:
                # no perp 2-folds
                pg = decide_pointgroup(
                    axeslist=[refined_mainax], orderlist=[mainax_bestorder]
                )[0]
        else:
            # no 2-folds
            pg = "C1"
    except:
        fobj.write(traceback.format_exc())
        pg = "C1"
    return pg


def justnfolds(emmap1, sordernax, sordernfsc, sordern, fobj):
    # only Cyclic symmetries possible
    pg = "C1"
    mainax = None
    try:
        if len(sordernax) > 0:
            # find the initial order for mainaxis
            for i, ax in enumerate(sordernax):
                refinement_results = refine_ax(
                    emmap1=emmap1,
                    axlist=[ax],
                    orderlist=[sordern[i]],
                    fobj=fobj,
                )
                if refinement_results is not None:
                    (
                        ref_axlist,
                        ref_tlist,
                        _,
                        _,
                        ref_fsclist,
                    ) = refinement_results
                    refined_mainax = ref_axlist[0]
                    refined_mainax_fsc = ref_fsclist[0]
                    refined_t = ref_tlist[0]
                    if refined_mainax_fsc >= pg_decide_fsc:
                        mainax = refined_mainax
                        mainax_number = i
                        mainax_order = sordern[i]
                        break

            if mainax is None:
                return pg

            # get t_centroid
            """ result = [
                abs(refined_t[i]*emmap1.map_dim[i]*emmap1.pix[i]) < 0.05 for i in range(3)]
            if all(result):
                t_centroid = refined_t
            else:
                t_centroid = get_t_to_centroid(
                    emmap1=emmap1, axis=refined_mainax, 
                    order=mainax_order) """
            t_centroid = refined_t

            # check for parellel axes
            par_nax_order = [mainax_order]
            for i, ax in enumerate(sordernax):
                if i != mainax_number:
                    emdalogger.log_string(
                        fobj,
                        "   Axis1: %s   Order: %s   FSC: % .3f"
                        % (
                            vec2string(refined_mainax),
                            mainax_order,
                            refined_mainax_fsc,
                        ),
                    )
                    emdalogger.log_string(
                        fobj,
                        "   Axis2: %s   Order: %s   FSC: % .3f"
                        % (vec2string(ax), sordern[i], sordernfsc[i]),
                    )
                    angle = cosine_angle(refined_mainax, ax)
                    # fobj.write(
                    #    '   Angle between them: % .3f\n' % angle)
                    emdalogger.log_string(
                        fobj, "   Angle between them: % .3f" % angle
                    )
                    # check if axis is parellel
                    if (
                        abs(angle) < ang_tol_p
                        or abs(angle - 180.0) < ang_tol_p
                    ):
                        par_nax_order.append(sordern[i])

            if len(par_nax_order) > 1:
                # mainax bestorder using parellel orders
                (mainax_bestorder, mainax_bestoder_fsc) = get_axorder(
                    emmap1=emmap1,
                    refined_axis=refined_mainax,
                    order_list=par_nax_order,
                    fobj=fobj,
                    t=t_centroid,
                )
            else:
                # mainax best order
                (mainax_bestorder, mainax_bestoder_fsc) = get_axorder(
                    emmap1=emmap1,
                    refined_axis=refined_mainax,
                    order_list=[mainax_order],
                    fobj=fobj,
                    t=t_centroid,
                )

            # decide pointgroup
            pg = decide_pointgroup(
                axeslist=[refined_mainax], orderlist=[mainax_bestorder]
            )[0]
    except:
        fobj.write(traceback.format_exc())
    return pg
