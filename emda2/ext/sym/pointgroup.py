
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import emda2.emda_methods2 as em
from emda2.ext.utils import (
    cut_resolution_for_linefit, 
    shift_density, 
    center_of_mass_density)
from emda2.ext.sym.filter_axes import cosine_angle


def decide_pointgroup(axeslist, orderlist):
    # check for cyclic sym of n-order
    order_arr = np.asarray(orderlist)
    dic = {i: (order_arr == i).nonzero()[0] for i in np.unique(order_arr)}
    uniqorder = np.fromiter(dic.keys(), dtype='int')
    anglist = []
    point_group = None
    gp_generator_ax1 = None
    gp_generator_ax2 = None
    order1 = order2 = 0
    ang_tol = 1.0  # Degrees
    print("unique orders: ", uniqorder)
    if len(uniqorder) == 1:
        print("Len of uniqorder: ", 1)
        if uniqorder[0] == 1:
            print("Unsymmetrized map")
            point_group = 'C1'
            order1 = 1
        elif uniqorder[0] != 1:
            odrn = dic[uniqorder[0]]
            if len(odrn) == 1:
                point_group = 'C'+str(uniqorder[0])
                gp_generator_ax1 = axeslist[0]
                order1 = uniqorder[0]
            elif len(odrn) > 1:
                if uniqorder[0] == 2:
                    print("Checking for D symmetry...")
                    for i in odrn:
                        for j in odrn:
                            if i != j:
                                angle = cosine_angle(axeslist[i], axeslist[j])
                                anglist.append(angle)
                    angarr = np.asarray(anglist, 'float')
                    if np.all(abs(angarr - 90.0) <= ang_tol):
                        point_group = 'D2'
                        gp_generator_ax1 = axeslist[odrn[0]]
                        gp_generator_ax2 = axeslist[odrn[1]]
                        order1 = order2 = 2
                    else:
                        print("Unknown symmetry")
                        point_group = 'Unkown'
                elif uniqorder[0] == 3:
                    print("Checking for T symmetry...")
                    for i in odrn:
                        for j in odrn:
                            if i != j:
                                angle = cosine_angle(axeslist[i], axeslist[j])
                                anglist.append(angle)
                    angarr = np.asarray(anglist, 'float')
                    condition2 = angarr[np.logical_not(
                        abs(angarr - 109.47) <= ang_tol)]
                    if np.all(abs(condition2 - 70.53) <= ang_tol):
                        point_group = 'T'
                        gp_generator_ax1 = axeslist[odrn[0]]
                        gp_generator_ax2 = axeslist[odrn[1]]
                        order1 = order2 = 3
                    else:
                        print("test1")
                        print("Unknown point group.")
                        point_group = 'Unkown'
                elif uniqorder[0] == 4:
                    print("Checking for O symmetry...")
                    for i in odrn:
                        for j in odrn:
                            if i != j:
                                angle = cosine_angle(axeslist[i], axeslist[j])
                                anglist.append(angle)
                    angarr = np.asarray(anglist, 'float')
                    if np.all(abs(angarr - 90.0) <= ang_tol):
                        point_group = 'O'
                        gp_generator_ax1 = axeslist[odrn[0]]
                        gp_generator_ax2 = axeslist[odrn[1]]
                        order1 = order2 = 4
                    else:
                        print("Unknown symmetry")
                        point_group = 'Unkown'
                elif uniqorder[0] == 5:
                    print("Checking for I symmetry...")
                    for i in odrn:
                        for j in odrn:
                            if i != j:
                                angle = cosine_angle(axeslist[i], axeslist[j])
                                anglist.append(angle)
                    angarr = np.asarray(anglist, 'float')
                    condition2 = angarr[np.logical_not(
                        abs(angarr - 63.47) <= ang_tol)]
                    if np.all(abs(condition2 - 116.57) <= ang_tol):
                        point_group = 'I'
                        gp_generator_ax1 = axeslist[odrn[0]]
                        gp_generator_ax2 = axeslist[odrn[1]]
                        order1 = order2 = 5
                    else:
                        print("Unknown symmetry")
                        point_group = 'Unkown'
                else:
                    print("Unknown symmetry")
                    point_group = 'Unkown'
    elif len(uniqorder) == 2:
        if np.any(uniqorder == 2):
            odr2 = dic[2]
            if np.any(uniqorder == 1):
                point_group = 'C2'
            elif np.any(uniqorder == 3):
                odr3 = dic[3]  # get all 3-fold axes locations
                if len(odr3) == 1:
                    print("Ckecking for D symmetry...")
                    for i in odr2:
                        for j in odr3:
                            angle = cosine_angle(axeslist[i], axeslist[j])
                            anglist.append(angle)
                    angarr = np.asarray(anglist, 'float')
                    if np.all(abs(angarr - 90.0) <= ang_tol):
                        point_group = 'D3'
                        gp_generator_ax1 = axeslist[odr2[0]]
                        gp_generator_ax2 = axeslist[odr3[0]]
                        order1 = 2
                        order2 = 3
                    else:
                        print("Unknown symmetry")
                        point_group = 'Unkown'
                if len(odr3) > 1:
                    print("Checking for T symmetry...")
                    for i in odr3:
                        for j in odr3:
                            if i != j:
                                angle = cosine_angle(axeslist[i], axeslist[j])
                                anglist.append(angle)
                    angarr = np.asarray(anglist, 'float')
                    #condition1 = angarr[abs(angarr - 109.47) <= ang_tol]
                    condition2 = angarr[np.logical_not(
                        abs(angarr - 109.47) <= ang_tol)]
                    if np.all(abs(condition2 - 70.53) <= ang_tol):
                        point_group = 'T'
                        gp_generator_ax1 = axeslist[odr3[0]]
                        gp_generator_ax2 = axeslist[odr3[1]]
                        order1 = order2 = 3
                    else:
                        print("test2")
                        print("Unknown point group.")
                        point_group = 'Unkown'
            elif np.any(uniqorder == 4):
                odr4 = dic[4]
                if len(odr4) == 1:
                    print("Checking for D symmetry...")
                    for i in odr2:
                        for j in odr4:
                            angle = cosine_angle(axeslist[i], axeslist[j])
                            anglist.append(angle)
                    angarr = np.asarray(anglist, 'float')
                    if np.all(abs(angarr - 90.0) <= ang_tol):
                        point_group = 'D4'
                        gp_generator_ax1 = axeslist[odr2[0]]
                        gp_generator_ax2 = axeslist[odr4[0]]
                        order1 = 2
                        order2 = 4
                    else:
                        print("Unknown symmetry")
                        point_group = 'Unkown'
                elif len(odr4) > 1:
                    print("Checking for O symmetry...")
                    for i in odr4:
                        for j in odr4:
                            if i != j:
                                angle = cosine_angle(axeslist[i], axeslist[j])
                                anglist.append(angle)
                    angarr = np.asarray(anglist, 'float')
                    if np.all(abs(angarr - 90.0) <= ang_tol):
                        point_group = 'O'
                        gp_generator_ax1 = axeslist[odr4[0]]
                        gp_generator_ax2 = axeslist[odr4[1]]
                        order1 = order2 = 4
                    else:
                        print("Unknown symmetry")
                        point_group = 'Unkown'
            elif np.any(uniqorder == 5):
                odr5 = dic[5]
                if len(odr5) == 1:
                    print("Checking for D symmetry...")
                    for i in odr2:
                        for j in odr5:
                            angle = cosine_angle(axeslist[i], axeslist[j])
                            anglist.append(angle)
                    angarr = np.asarray(anglist, 'float')
                    if np.all(abs(angarr - 90.0) <= ang_tol):
                        point_group = 'D5'
                        gp_generator_ax1 = axeslist[odr2[0]]
                        gp_generator_ax2 = axeslist[odr5[0]]
                        order1 = 2
                        order2 = 5
                    else:
                        print("Unknown symmetry")
                        point_group = 'Unkown'
                if len(odr5) > 1:
                    print("Checking for I symmetry...")
                    for i in odr5:
                        for j in odr5:
                            if i != j:
                                angle = cosine_angle(axeslist[i], axeslist[j])
                                anglist.append(angle)
                    angarr = np.asarray(anglist, 'float')
                    condition2 = angarr[np.logical_not(
                        abs(angarr - 63.47) <= ang_tol)]
                    if np.all(abs(condition2 - 116.57) <= ang_tol):
                        point_group = 'I'
                        gp_generator_ax1 = axeslist[odr5[0]]
                        gp_generator_ax2 = axeslist[odr5[1]]
                        order1 = order2 = 5
                    else:
                        print("Unknown symmetry")
                        point_group = 'Unkown'
            else:
                n = uniqorder[uniqorder != 2][0]
                odrn = dic[n]
                print("Ckecking for D symmetry...")
                for i in odr2:
                    for j in odrn:
                        angle = cosine_angle(axeslist[i], axeslist[j])
                        anglist.append(angle)
                angarr = np.asarray(anglist, 'float')
                if np.all(abs(angarr - 90.0) <= ang_tol):
                    point_group = 'D' + str(n)
                    gp_generator_ax1 = axeslist[odr2[0]]
                    gp_generator_ax2 = axeslist[odrn[0]]
                    order1 = 2
                    order2 = n
                else:
                    print("Unknown symmetry")
                    point_group = 'Unkown'
    elif len(uniqorder) > 2:
        # groups must belong to O or I
        if np.any(uniqorder == 2):
            odr2 = dic[2]
            if np.any(uniqorder == 3):
                odr3 = dic[3]
                if np.any(uniqorder == 4):
                    odr4 = dic[4]
                    print("Ckecking for O symmetry...")
                    for i in odr4:
                        for j in odr4:
                            if i != j:
                                angle = cosine_angle(axeslist[i], axeslist[j])
                                anglist.append(angle)
                    angarr = np.asarray(anglist, 'float')
                    if np.all(abs(angarr - 90.0) <= ang_tol):
                        point_group = 'O'
                        gp_generator_ax1 = axeslist[odr4[0]]
                        gp_generator_ax2 = axeslist[odr4[1]]
                        order1 = order2 = 4
                    else:
                        print("Unknown point group.")
                        point_group = 'Unkown'
                elif np.any(uniqorder == 5):
                    # I symmetry
                    odr5 = dic[5]
                    print("Ckecking for I symmetry...")
                    for i in odr5:
                        for j in odr5:
                            if i != j:
                                angle = cosine_angle(axeslist[i], axeslist[j])
                                anglist.append(angle)
                    angarr = np.asarray(anglist, 'float')
                    #condition1 = angarr[abs(angarr - 63.47) <= ang_tol]
                    condition2 = angarr[np.logical_not(
                        abs(angarr - 63.47) <= ang_tol)]
                    if np.all(abs(condition2 - 116.57) <= ang_tol):
                        point_group = 'I'
                        gp_generator_ax1 = axeslist[odr5[0]]
                        gp_generator_ax2 = axeslist[odr5[1]]
                        order1 = order2 = 5
                    else:
                        print("Unknown point group.")
                        point_group = 'Unkown'
        else:
            print("Unknown symmetry")
            point_group = 'Unkown'
    return point_group, [order1, order2, gp_generator_ax1, gp_generator_ax2]


def get_pg(axis_list, orderlist, m1, resol=5.0, fsc_cutoff=0.7, fobj=None):
    nbin, res_arr, bin_idx, _ = em.get_binidx(cell=m1.workcell, arr=m1.workarr)
    dist = np.sqrt((res_arr - resol) ** 2)
    cbin = np.argmin(dist) + 1
    # using COM
    nx, ny, nz = m1.workarr.shape
    com = center_of_mass_density(m1.workarr)
    print("com:", com)
    box_centr = (nx // 2, ny // 2, nz // 2)
    arr = shift_density(m1.workarr, np.subtract(box_centr, com))
    m1.workarr = arr
    #       
    fo = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(m1.workarr)))
    fcut, cBIdx, cbin = cut_resolution_for_linefit(
        [fo], bin_idx, res_arr, cbin
    )
    tlist = [[0., 0., 0.] for _ in range(len(axis_list))]
    # initial filtering of axes - remove duplicates
    from emda2.ext.sym import filter_axes
    results = filter_axes.filter_axes(axlist=axis_list,
                          orderlist=orderlist,
                          fo=fcut[0,:,:,:],
                          bin_idx=cBIdx,
                          nbin=cbin,
                          fsc_cutoff=0.0,
                          fsc_tol=0.1,
                          fobj=fobj,
                          tlist=tlist)
    if len(results) < 1:
        return []
    cleaned_axlist, cleaned_odrlist = results[0], results[1]
    # refine axes
    refax_list = []
    reft_list = []
    from emda2.ext import axis_refinement
    emmap1 = axis_refinement.EmmapOverlay(arr=m1.workarr)
    emmap1.bin_idx = cBIdx
    emmap1.res_arr = res_arr[:cbin]
    emmap1.nbin = cbin
    emmap1.fo_lst = [fcut[0,:,:,:]]
    t_init = [0.0, 0.0, 0.0]
    #if fobj is not None:
    #    fobj.write("resolution   angle   axini   FSCini   axfnl   FSCfnl\n")
    for ax, order in zip(cleaned_axlist, cleaned_odrlist):
        ax_ini, ax_fnl, t_fnl = axis_refinement.axis_refine(
            emmap1=emmap1,
            rotaxis=ax,
            symorder=order,
            fitres=resol,
            fitfsc=0.0,#0.5,
            ncycles=10,
            fobj=fobj,
            t_init=t_init,
        )
        refax_list.append(ax_fnl)
        reft_list.append(t_fnl)
    # filter axes based on FSC
    results = filter_axes.filter_axes(axlist=refax_list,
                          orderlist=cleaned_odrlist,
                          fo=fcut[0,:,:,:],
                          bin_idx=cBIdx,
                          nbin=cbin,
                          fsc_cutoff=fsc_cutoff,
                          fsc_tol=0.01,
                          fobj=fobj,
                          tlist=reft_list
                          )
    if len(results) > 0:
        cleaned_axlist, cleaned_odrlist = results[0], results[1]
        pg, gp_generators = decide_pointgroup(
            axeslist=cleaned_axlist, orderlist=cleaned_odrlist)
        print("Point group detected from the map: ", pg)
        gen_axlist = [gp_generators[2], gp_generators[3]]
        gen_odrlist = [gp_generators[0], gp_generators[1]]
        return [pg, gen_odrlist, gen_axlist]
    else:
        return []
