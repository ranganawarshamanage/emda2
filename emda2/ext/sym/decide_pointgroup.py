"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

import numpy as np
from emda2.ext.sym.filter_axes import cosine_angle
from more_itertools import sort_together

def decide_pointgroup(axeslist, orderlist):
    # check for cyclic sym of n-order
    order_arr = np.asarray(orderlist)
    dic = {i: (order_arr == i).nonzero()[0] for i in np.unique(order_arr)}
    uniqorder = np.fromiter(dic.keys(), dtype='int')
    anglist = []
    pg = 'C1'
    gp_generator_ax1 = None
    gp_generator_ax2 = None
    order1 = order2 = 0
    ang_tol = 1.0  # Degrees

    if len(axeslist) == 0:
        return ['C1']

    if len(axeslist) == 1:
        # options - Cyclic symmetries
        pg = 'C' + str(orderlist[0])
        return [pg]

    if len(axeslist) == 2:
        # options - I, O, T, D
        angle = cosine_angle(axeslist[0], axeslist[1])
        #print('angle: ', angle)
        if (abs(angle - 63.47) <= ang_tol or 
                abs(angle - 116.57) <= ang_tol):
            assert orderlist[0] == orderlist[1] == 5
            pg = 'I'
        elif abs(angle - 90.0) < ang_tol:
            # options - O, D
            if orderlist[0] == orderlist[1] == 4:
                pg = 'O'
            else:
                assert min(orderlist) == 2
                pg = 'D'+str(max(orderlist))
        elif abs(angle - 35.26) <= ang_tol or abs(angle - 144.74) <= ang_tol:
            if min(orderlist) == 2 and max(orderlist) == 3:
                pg = 'O'
        elif (abs(angle - 109.47) <= ang_tol or 
                abs(angle - 70.53) <= ang_tol):
            assert max(orderlist) == 3
            assert min(orderlist) == 3
            pg = 'T'
        elif (abs(angle - 54.74) <= ang_tol or 
                abs(angle - 125.26) <= ang_tol):
            if max(orderlist) == 3 and min(orderlist) == 2:
                pg = 'T'   
            if max(orderlist) == 4 and min(orderlist) == 3:
                pg = 'O'                     
        elif angle < ang_tol or (180. - angle) < ang_tol:
            if max(orderlist) % min(orderlist) == 0:
                pg = 'C' + str(max(orderlist))
            else:
                pg = 'C' + str(orderlist[0] * orderlist[1])
        else:
            pg = 'Unknown'
        return [pg]

    if len(axeslist) > 2:
        orderlist, axeslist = sort_together([orderlist, axeslist], reverse=True)
        
        # check for I sym
        ax5 = []
        for i, order in enumerate(orderlist):
            if order == 5:
                ax5.append(axeslist[i])
        if len(ax5) >= 2:
            angle = cosine_angle(ax5[0], ax5[1])
            if (abs(angle - 63.47) <= ang_tol or 
                abs(angle - 116.57) <= ang_tol):
                pg = 'I'
                return [pg]
        
        # check for O sym
        ax4 = []
        for i, order in enumerate(orderlist):
            if order == 4:
                ax4.append(axeslist[i])
        if len(ax4) >= 2:
            angle = cosine_angle(ax4[0], ax4[1])
            if abs(angle - 90.0) < ang_tol:
                pg = 'O'
                return [pg]

        # check for T sym
        ax2, ax3 = [], []
        for i, order in enumerate(orderlist):
            if order == 3:
                ax3.append(axeslist[i])
            elif order == 2:
                ax2.append(axeslist[i])
        if len(ax3) > 0 and len(ax2) > 0:
            for axis1 in ax3:
                for axis2 in ax2:
                    angle = cosine_angle(axis1, axis2)
                    if (abs(angle - 54.74) <= ang_tol or 
                        abs(angle - 125.26) <= ang_tol):
                        pg = 'T'
                        return [pg]  
                    else:
                        continue
        if len(ax3) >= 2:
            angle = cosine_angle(ax3[0], ax3[1])
            if (abs(angle - 109.47) <= ang_tol or 
                abs(angle - 70.53) <= ang_tol):
                pg = 'T'
                return [pg]

        # check for D sym
        ax2 = []
        for i, order in enumerate(orderlist):
            if order == 2:
                ax2.append(axeslist[i])
        if len(ax2) > 0:
            temp_order_list = []
            for i, order in enumerate(orderlist):
                if order > 2:
                    angle = cosine_angle(axeslist[i], ax2[0])
                    if abs(angle - 90.0) < ang_tol:
                        temp_order_list.append(order)
            if len(temp_order_list) > 0:
                maxorder = max(temp_order_list)
                pg = 'D' + str(maxorder)
                return [pg]
        if len(ax2) >= 2:
            angle = cosine_angle(ax2[0], ax2[1])
            if abs(angle - 90.0) < ang_tol:
                pg = 'D' + str(orderlist[0])
                return [pg]

        # check for C sym
        ax1 = axeslist[0]
        mask = [True]
        for i, ax in enumerate(axeslist[1:]):
            angle = cosine_angle(ax1, ax)
            if angle < ang_tol or (180. - angle) < ang_tol:
                mask.append(True)
            else:
                mask.append(False)
        if all(mask):
            order = max(orderlist)
            if order >= 360:
                pg = 'Ukwn'
            else:
                pg = 'C%s'%order
        else:
            pg = 'Unknown'
        return [pg]   


if __name__=="__main__":
    axlist = [
        [0.000, -0.000,  1.000],
        [0.000,  0.816, -0.577],
    ]

    orderlist = [3, 2]

    print(decide_pointgroup(axlist, orderlist)[0])