import numpy as np
import math
from more_itertools import sort_together
from emda2.core import fsctools, maptools


def prime_factors(n):
    # https://stackoverflow.com/questions/15347174/python-finding-prime-factors
    i = 2
    factors = []
    #factors.append(2)
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def prefilter_order(f1, axis, order, nbin, bin_idx):
    # prime factorization
    factors = prime_factors(order)
    factors.append(np.prod(np.asarray(factors), dtype='int'))
    avgfsclist = []
    true_order = None
    if len(factors) == 1:
        if factors[0] != 2:
            factors.append(2)
        else:
            true_order = 2
    if true_order is None:
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        for fac in factors:
            theta = float(360.0 / fac)
            f_transformed = maptools.map_transform(
                flist=[f1], 
                axis=axis, 
                translation=[0., 0., 0.], 
                angle=theta
                )
            fsc = fsctools.anytwomaps_fsc_covariance(
                f1, f_transformed[0], bin_idx, nbin)[0]
            avg_fsc = np.average(fsc)
            avgfsclist.append(avg_fsc)
        true_order = factors[np.argmax(np.asarray(avgfsclist))]
    return true_order

def cosine_angle(ax1, ax2):
    vec_a = np.asarray(ax1, 'float')
    vec_b = np.asarray(ax2, 'float')
    dotp = np.dot(vec_a, vec_b)
    if -1.0 <= dotp <= 1.0:
        angle = math.acos(dotp)
    else:
        print("Problem, dotp: ", dotp)
        print("axes:", vec_a, vec_b)
        angle = 0.0
    return np.rad2deg(angle)


def get_avgfsc(order, axis, f1, bin_idx, nbin):
    theta = float(360.0 / order)
    f_transformed = maptools.map_transform(
        flist=[f1], 
        axis=axis, 
        translation=[0., 0., 0.], 
        angle=theta
        )
    fsc = fsctools.anytwomaps_fsc_covariance(
        f1, f_transformed[0], bin_idx, nbin)[0]
    return np.average(fsc)


def filter_axes(axlist, orderlist, fo, bin_idx, nbin, 
        fsc_cutoff=0.0, ang_tol=1.0, fsc_tol=0.2, fobj=None, tlist=None):
    """
    Inputs:
        axlist: axislist from proshade.get_symmops_from_proshade
        orderlist: orderlist from proshade.get_symmops_from_proshade
        arr: density (ndarray)
    """
    cleaned_axlist = []
    cleaned_odrlist = []
    cleaned_tlist = []
    duplicate_axes_list = []
    duplicate_order_list = [] 
    if len(orderlist) < 1:
        print("Empty list!")
        return []
    elif len(orderlist) == 1:
        true_order = prefilter_order(
            f1=fo,
            axis=axlist[0], 
            order=orderlist[0],
            bin_idx=bin_idx,
            nbin=nbin)
        cleaned_tlist.append(tlist[0])
        cleaned_axlist.append(axlist[0])
        cleaned_odrlist.append(true_order) 
    else:
        # sort all lists by order
        sorted_list = sort_together([orderlist, axlist, tlist])
        sorted_odrlist, sorted_axlist, sorted_tlist = sorted_list
        # remove duplicate axes
        duplicate_list = []
        for i, odr1 in enumerate(sorted_odrlist):
            for j, odr2 in enumerate(sorted_odrlist):
                if i < j:
                    angle = cosine_angle(sorted_axlist[i], sorted_axlist[j])
                    if angle < ang_tol or (180. - angle) < ang_tol:
                        if odr1 == odr2:
                            duplicate_list.append(j)
                        else:
                            afsc1 = get_avgfsc(order=sorted_odrlist[i], 
                                axis=sorted_axlist[i], 
                                f1=fo,
                                bin_idx=bin_idx,
                                nbin=nbin)
                            afsc2 = get_avgfsc(order=sorted_odrlist[j], 
                                axis=sorted_axlist[j], 
                                f1=fo,
                                bin_idx=bin_idx,
                                nbin=nbin)
                            print('ax, order1, order2, afsc1, afsc2: ', sorted_axlist[i], odr1, odr2, afsc1, afsc2)
                            if abs(afsc1 - afsc2) <= fsc_tol:
                                if odr1 > odr2:
                                    duplicate_list.append(j)
                                else:
                                    duplicate_list.append(i) 
                            else:
                                if afsc1 > afsc2:
                                    duplicate_list.append(j)
                                else:
                                    duplicate_list.append(i)                         
        for i, ax in enumerate(sorted_axlist):
            if not np.any(i == np.asarray(duplicate_list)):
                # true_order by prime factorization
                true_order = prefilter_order(
                    f1=fo,
                    axis=ax, 
                    order=sorted_odrlist[i],
                    bin_idx=bin_idx,
                    nbin=nbin)
                cleaned_tlist.append(sorted_tlist[i])
                cleaned_axlist.append(ax)
                cleaned_odrlist.append(true_order)
            else:
                duplicate_axes_list.append(ax)
                duplicate_order_list.append(sorted_odrlist[i])
    print('Number of duplicate axes: ', len(duplicate_axes_list))
    if len(duplicate_axes_list) > 0:
        print("Duplicate axes:")
        for ax, order in zip(duplicate_axes_list, duplicate_order_list):
            print(ax, order)
    print('Number of unique axes: ', len(cleaned_axlist))
    if len(cleaned_axlist) > 0:
        print("Unique Axes:")
        for i, ax in enumerate(cleaned_axlist):
            print(ax, cleaned_odrlist[i])
    # Now calculate FSC per axis
    assert len(cleaned_axlist) > 0
    axes_list = []
    order_list = []
    avgfsc_list = []
    print("Filtering by FSC...")
    print('Axes with avg_fsc >= %s:' %(fsc_cutoff))
    for i, axis in enumerate(cleaned_axlist):
        order = cleaned_odrlist[i]
        axis = np.asarray(axis, 'float')
        axis = axis / math.sqrt(np.dot(axis, axis))
        theta = float(360.0 / order)
        f_transformed = maptools.map_transform(
            flist=[fo], 
            axis=axis, 
            translation=cleaned_tlist[i], 
            angle=theta
            )
        fsc = fsctools.anytwomaps_fsc_covariance(
            fo, f_transformed[0], bin_idx, nbin)[0]
        avg_fsc = np.average(fsc)
        if avg_fsc > fsc_cutoff:
            axes_list.append(axis)
            order_list.append(order)
            avgfsc_list.append(avg_fsc)
            print('axis=',axis, 'order=',order, 'FSC=',avg_fsc)
    if len(avgfsc_list) > 0:
        # sort axes list by FSC
        sorted_list = sort_together([avgfsc_list, axes_list, order_list], reverse=True)
        sorted_fsclist, sorted_axlist, sorted_orderlist = sorted_list 
        return [sorted_axlist, sorted_orderlist]
    else:
        return []

