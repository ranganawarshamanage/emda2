from __future__ import absolute_import, division, print_function, unicode_literals
import enum
import itertools
from timeit import default_timer as timer
import numpy as np
import math
from numpy.fft import fftn, ifftn, fftshift, ifftshift
# EMDA2 imports
from emda2.core import quaternions, restools, plotter, iotools
from emda2.ext.utils import (
    rotate_f, 
    shift_density, 
    center_of_mass_density, 
    cut_resolution_for_linefit,
    determine_ibin,
    get_ibin,
    filter_fsc,
    vec2string
)
import fcodes2 as fc
from emda2.core import fsctools

class EmmapOverlay:
    def __init__(self, arr):
        self.arr = arr
        self.map_unit_cell = None
        self.map_dim = arr.shape
        self.com = True
        self.com1 = None
        self.com2 = None
        self.box_centr = None
        self.nbin = None
        self.res_arr = None
        self.bin_idx = None
        self.fo_lst = []
        self.pix = None

    def prep_data(self):
        com = center_of_mass_density(self.arr)
        print("com:", com)
        nx, ny, nz = self.arr.shape
        self.box_centr = (nx // 2, ny // 2, nz // 2)
        self.com1 = com
        if self.com:
            arr = shift_density(self.arr, np.subtract(self.box_centr, com))
            self.com2 = center_of_mass_density(arr)
            print("com after centering:", self.com2)
            self.fo_lst.append(fftshift(fftn(fftshift(arr))))
        else:
            self.fo_lst.append(fftshift(fftn(fftshift(self.arr))))
        if any(itm is None for itm in [self.bin_idx, self.nbin, self.res_arr]):
            if self.map_unit_cell is not None:
                self.pix = [self.map_unit_cell[i]/sh for i, sh in enumerate(self.map_dim)]
                self.nbin, self.res_arr, self.bin_idx, _ = restools.get_resolution_array(
                    self.map_unit_cell, self.fo_lst[0]
                )
            else:
                print("make sure cell is included")


class Bfgs:
    def __init__(self):
        self.method = "BFGS"
        self.e0 = None
        self.e1 = None
        self.ax_init = None
        self.ax_final = None
        self.angle = None
        self.q = np.array([1., 0., 0., 0.], 'float')
        self.t = np.array([0., 0., 0.], 'float')
        self.xyz = None
        self.xyz_sum = None
        self.vol = None
        self.bin_idx = None
        self.binfsc = None
        self.nbin = None
        self.x = np.array([0.1, 0.1, 0.1], 'float')

    def calc_fsc(self,e1):
        #print("FSC calculation")
        assert self.e0.shape == e1.shape == self.bin_idx.shape
        self.binfsc = fsctools.anytwomaps_fsc_covariance(
            self.e0, e1, self.bin_idx, self.nbin)[0]

    def get_wght(self, e1): 
        self.calc_fsc(e1)
        #print("weight calculation")
        nz, ny, nx = self.e0.shape
        val_arr = np.zeros((self.nbin, 2), dtype='float')
        val_arr[:,0] = self.binfsc #/ (1 - self.binfsc ** 2)
        self.wgrid = fc.read_into_grid(self.bin_idx,
            self.binfsc, self.nbin, nz, ny, nx)

    def functional(self, x, info):
        #print("functional")
        nx, ny, nz = self.e0.shape
        ax = self.ax_init + x[:3]
        ax = ax / math.sqrt(np.dot(ax, ax))
        q = quaternions.get_quaternion(axis=ax, angle=self.angle)
        q = q / np.sqrt(np.dot(q, q))
        rotmat = quaternions.get_RM(q)
        e1 = rotate_f(rotmat, self.e0, interp='linear')[:, :, :, 0]
        t = x[3:]
        st, _, _, _ = fc.get_st(nx, ny, nz, t)
        e1 = e1 * st
        self.get_wght(e1)
        fval = np.sum(self.wgrid * self.e0 * np.conjugate(e1)) / (nx*ny*nz) # divide by vol is to scale
        #fval = np.sum(self.e0 * np.conjugate(e1)) / (nx*ny*nz)
        if info['Nfeval'] % 20 == 0:
            print('fval, axis, trans', fval.real, ax, t)
        info['Nfeval'] += 1
        return -fval.real
 
    def optimize(self):
        from scipy.optimize import minimize
        x = np.array([0.0, 0.0, 0.0, 0., 0., 0.], 'float')
        options = {'maxiter': 2000}
        args=({'Nfeval':0},)
        print("Optimization method: ", self.method)
        if self.method.lower() == 'nelder-mead':
            result = minimize(
                fun=self.functional, 
                x0=x, 
                method='Nelder-Mead', 
                tol=1e-5, 
                options=options,  
                args=args
                )  
        if result.status:
            print(result)
        self.t = result.x[3:]
        ax = self.ax_init + result.x[:3]
        self.ax_final = ax / math.sqrt(np.dot(ax, ax))
        print('Final axis: ', self.ax_final)   


def fsc_between_static_and_transfomed_map(
    staticmap, movingmap, bin_idx, rm, t, cell, nbin
):
    nx, ny, nz = staticmap.shape
    st, _, _, _ = fc.get_st(nx, ny, nz, t)
    frt_full = rotate_f(rm, movingmap * st, interp="linear")[:, :, :, 0]
    f1f2_fsc = fsctools.anytwomaps_fsc_covariance(
        staticmap, frt_full, bin_idx, nbin)[0]
    return f1f2_fsc

#def run_fit(
#    emmap1,
#    rotmat,
#    t,
#    ifit=0,
#    fitfsc=0.5,
#    nmarchingcycles=10,
#    fobj=None,
#    fitres=None,
#):
#    q_init = quaternions.rot2quart(rotmat)
#    axis_ang = quaternions.quart2axis(q_init)
#    axis_ini = axis_ang[:3]
#    angle = axis_ang[-1]
#    if fitres is not None:
#        if fitres <= emmap1.res_arr[-1]:
#            fitbin = len(emmap1.res_arr) - 1
#        else:
#            dist = np.sqrt((emmap1.res_arr - fitres) ** 2)
#            ibin = np.argmin(dist)
#            if ibin % 2 != 0:
#                ibin = ibin - 1
#            fitbin = min([len(dist), ibin])
#    if fitres is None:
#        fitbin = len(emmap1.res_arr) - 1
#    fsc_lst = []
#    for i in range(nmarchingcycles):
#        if i == 0:
#            f1f2_fsc = fsc_between_static_and_transfomed_map(
#                staticmap=emmap1.fo_lst[0],
#                movingmap=emmap1.fo_lst[0],
#                bin_idx=emmap1.bin_idx,
#                rm=rotmat,
#                t=t,
#                cell=emmap1.map_unit_cell,
#                nbin=emmap1.nbin,
#            )
#            fsc_lst.append(f1f2_fsc)
#            if fitfsc > 0.999:
#                rotmat = rotmat
#                final_axis = axis_ini
#                print("\n***FSC between static and moving maps***\n")
#                print("bin#     resolution(A)      start-FSC     end-FSC\n")
#                for j in range(len(emmap1.res_arr)):
#                    print(
#                        "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
#                            j, emmap1.res_arr[j], fsc_lst[0][j], fsc_lst[0][j]
#                        )
#                    )
#                break
#            ibin = determine_ibin(bin_fsc=f1f2_fsc)
#            if fitbin < ibin:
#                ibin = fitbin
#            ibin_old = ibin
#            q = q_init
#            print("Fitting starts at ", emmap1.res_arr[ibin], " (A)")
#        else:
#            # Apply initial rotation and translation to calculate fsc
#            f1f2_fsc = fsc_between_static_and_transfomed_map(
#                emmap1.fo_lst[0],
#                emmap1.fo_lst[ifit],
#                emmap1.bin_idx,
#                rotmat,
#                t,
#                emmap1.map_unit_cell,
#                emmap1.nbin,
#            )          
#            ibin = get_ibin(filter_fsc(f1f2_fsc), cutoff=fitfsc)
#            if fitbin < ibin:
#                ibin = fitbin
#            if ibin_old == ibin:
#                fsc_lst.append(f1f2_fsc)
#                res_arr = emmap1.res_arr[:ibin_old]
#                fsc_bef = fsc_lst[0][:ibin_old]
#                fsc_aft = fsc_lst[1][:ibin_old]
#                print("\n***FSC between static and moving maps***\n")
#                print("bin#     resolution(A)      start-FSC     end-FSC\n")
#                for j in range(len(res_arr)):
#                    print(
#                        "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
#                            j, res_arr[j], fsc_bef[j], fsc_aft[j]
#                        )
#                    )
#                print("Plotting FSCs...")
#                plotter.plot_nlines(
#                    res_arr=res_arr, 
#                    list_arr=[fsc_lst[0][:ibin_old], fsc_lst[1][:ibin_old]], 
#                    curve_label=["Proshade axis", "EMDA axis"], 
#                    plot_title="FSC based on Symmetry axis", 
#                    fscline=1.,
#                    mapname="fsc_axis.eps")
#                break
#            else:
#                ibin_old = ibin
#                print("Fitting starts at ", emmap1.res_arr[ibin], " (A)")
#        if ibin == 0:
#            print("ibin = 0, Cannot proceed! Stopping current axis refinement.")
#            fobj.write(
#                "ibin = 0, Cannot proceed! Stopping current axis refinement.\n")
#            break
#        e_list = [emmap1.fo_lst[0]]
#        fcut, cBIdx, cbin = cut_resolution_for_linefit(
#            e_list, emmap1.bin_idx, emmap1.res_arr, ibin
#        )
#        static_cutmap = fcut[0, :, :, :]  # use Fo for fitting.
#        # test output for debug
#        """ nx, ny, nz = static_cutmap.shape
#        print('t: ', t)
#        st, _, _, _ = fc.get_st(nx, ny, nz, t)
#        st = 1.
#        moving_map = rotate_f(rotmat, static_cutmap * st, interp="linear")[:, :, :, 0]
#        stmap = np.real(ifftshift(ifftn(ifftshift(static_cutmap))))
#        mvmap = np.real(ifftshift(ifftn(ifftshift(moving_map))))
#        ms = iotools.Map(name='staticmap.mrc')
#        ms.arr = stmap
#        ms.cell = emmap1.map_unit_cell
#        ms.write()
#        mm = iotools.Map(name='movingmap.mrc')
#        mm.arr = mvmap
#        mm.cell = emmap1.map_unit_cell
#        mm.write()  """       
#        #
#        bfgs = Bfgs()
#        if i == 0:
#            bfgs.ax_init = np.asarray(axis_ini, 'float')
#        else:
#            bfgs.ax_init = current_axis
#        bfgs.angle = float(np.rad2deg(angle))
#        bfgs.e0 = static_cutmap
#        bfgs.bin_idx = cBIdx
#        bfgs.nbin = cbin
#        bfgs.method = 'nelder-mead'
#        bfgs.optimize()
#        current_axis = bfgs.ax_final
#        t = -bfgs.t
#        q = quaternions.get_quaternion(list(current_axis), bfgs.angle)
#        rotmat = quaternions.get_RM(q)
#    final_axis = current_axis
#    final_t = t
#    return final_axis, final_t

def run_fit(
    emmap1,
    rotmat,
    t,
    ifit=0,
    fitfsc=0.5,
    nmarchingcycles=10,
    fobj=None,
    fitres=None,
):
    q_init = quaternions.rot2quart(rotmat)
    axis_ang = quaternions.quart2axis(q_init)
    axis_ini = axis_ang[:3]
    angle = axis_ang[-1]
    if fitres is not None:
        if fitres <= emmap1.res_arr[-1]:
            fitbin = len(emmap1.res_arr) - 1
        else:
            dist = np.sqrt((emmap1.res_arr - fitres) ** 2)
            ibin = np.argmin(dist)
            if ibin % 2 != 0:
                ibin = ibin - 1
            fitbin = min([len(dist), ibin])
    if fitres is None:
        fitbin = len(emmap1.res_arr) - 1
    fsc_lst = []
    is_abandon = False
    final_axis = axis_ini
    final_t = t
    try:
        for i in range(nmarchingcycles):
            if i == 0:
                f1f2_fsc = fsc_between_static_and_transfomed_map(
                    staticmap=emmap1.fo_lst[0],
                    movingmap=emmap1.fo_lst[0],
                    bin_idx=emmap1.bin_idx,
                    rm=rotmat,
                    t=t,
                    cell=emmap1.map_unit_cell,
                    nbin=emmap1.nbin,
                )
                f1f2_fsc_old = f1f2_fsc
                fsc_lst.append(f1f2_fsc)
                if fitfsc > 0.999:
                    afsc_fnl = afsc_ini = np.average(f1f2_fsc[:fitbin])
                    resol_fsc = emmap1.res_arr[fitbin]
                    rotmat = rotmat
                    print("\n***FSC between static and moving maps***\n")
                    print("bin#     resolution(A)      start-FSC     end-FSC\n")
                    for j in range(len(emmap1.res_arr)):
                        print(
                            "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
                                j, emmap1.res_arr[j], fsc_lst[0][j], fsc_lst[0][j]
                            )
                        )
                    break
                ibin = determine_ibin(bin_fsc=f1f2_fsc)
                ibin_old = ibin
                if ibin >= 5:
                    if fitbin < ibin:
                        ibin = fitbin
                    ibin_old = ibin
                    q = q_init
                    print("Fitting starts at ", emmap1.res_arr[ibin], " (A)")
                else:
                    print('Resolution= %s (A)' %emmap1.res_arr[ibin])
                    print('FSC between two copies is too low. FSC= %s ibin=%s' %(f1f2_fsc[ibin], ibin))
            else:
                f1f2_fsc = fsc_between_static_and_transfomed_map(
                    emmap1.fo_lst[0],
                    emmap1.fo_lst[ifit],
                    emmap1.bin_idx,
                    rotmat,
                    t,
                    emmap1.map_unit_cell,
                    emmap1.nbin,
                )          
                ibin = get_ibin(filter_fsc(f1f2_fsc), cutoff=fitfsc)
                if fitbin < ibin:
                    ibin = fitbin
                if ibin_old == ibin:
                    fsc_lst.append(f1f2_fsc)
                    res_arr = emmap1.res_arr[:ibin]
                    fsc_bef = fsc_lst[0][:ibin]
                    fsc_aft = fsc_lst[1][:ibin]
                    print("\n***FSC between static and moving maps***\n")
                    print("bin#     resolution(A)      start-FSC     end-FSC\n")
                    for j in range(len(res_arr)):
                        print(
                            "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
                                j, res_arr[j], fsc_bef[j], fsc_aft[j]
                            )
                        )
                    print("Plotting FSCs...")
                    plotter.plot_nlines(
                        res_arr=res_arr, 
                        list_arr=[fsc_lst[0][:ibin_old], fsc_lst[1][:ibin_old]], 
                        curve_label=["Proshade axis", "EMDA axis"], 
                        plot_title="FSC based on Symmetry axis", 
                        fscline=1.,
                        mapname="fsc_axis.eps")
                    break
                elif ibin_old > ibin:
                    fsc_lst.append(f1f2_fsc_old)
                    res_arr = emmap1.res_arr[:ibin_old]
                    fsc_bef = fsc_lst[0][:ibin_old]
                    fsc_aft = fsc_lst[1][:ibin_old]
                    afsc_ini = np.average(fsc_bef)
                    afsc_fnl = np.average(fsc_aft)
                    resol_fsc = emmap1.res_arr[ibin_old]
                    print("\n***FSC between static and moving maps***\n")
                    print("bin#     resolution(A)      start-FSC     end-FSC\n")
                    for j in range(len(res_arr)):
                        print(
                            "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
                                j, res_arr[j], fsc_bef[j], fsc_aft[j]
                            )
                        )
                    print("Plotting FSCs...")
                    plotter.plot_nlines(
                        res_arr=res_arr, 
                        list_arr=[fsc_lst[0][:ibin_old], fsc_lst[1][:ibin_old]], 
                        curve_label=["Proshade axis", "EMDA axis"], 
                        plot_title="FSC based on Symmetry axis", 
                        fscline=1.,
                        mapname="fsc_axis.eps")
                    final_axis = final_axis_previous
                    final_t = final_t_previous
                    break
                else:
                    ibin_old = ibin
                    f1f2_fsc_old = f1f2_fsc
                    print("Fitting starts at ", emmap1.res_arr[ibin], " (A)")
            if ibin >= 5:
                e_list = [emmap1.fo_lst[0]]
                fcut, cBIdx, cbin = cut_resolution_for_linefit(
                    e_list, emmap1.bin_idx, emmap1.res_arr, ibin
                )
                static_cutmap = fcut[0, :, :, :]  # use Fo for fitting.
                bfgs = Bfgs()
                if i == 0:
                    bfgs.ax_init = np.asarray(axis_ini, 'float')
                else:
                    bfgs.ax_init = current_axis
                bfgs.angle = float(np.rad2deg(angle))
                bfgs.e0 = static_cutmap
                bfgs.bin_idx = cBIdx
                bfgs.nbin = cbin
                bfgs.method = 'nelder-mead'
                bfgs.optimize()
                current_axis = bfgs.ax_final
                t = -bfgs.t
                q = quaternions.get_quaternion(list(current_axis), bfgs.angle)
                rotmat = quaternions.get_RM(q)
                final_axis_previous = final_axis
                final_axis = current_axis
                final_t_previous = final_t
                final_t = t
            else:
                is_abandon = True
                print("ibin = %s, Cannot proceed axis refinement." %(ibin))
                fobj.write("ibin = %s, Cannot proceed axis refinement. \n" %(ibin))
                final_axis = []
                final_t = []
                pos_ax = []
                break
        if fobj is not None and not is_abandon:
            #fobj.write('   Refined axis: %s   Order: %s   FSC: % .3f @ % .2f A\n' %
            #    (vec2string(final_axis), int(360/np.rad2deg(angle)), afsc_fnl, resol_fsc))
            if emmap1.com:
                if emmap1.com1 is None:
                    emmap1.com1 = center_of_mass_density(emmap1.arr)
                pos_ax = [(emmap1.com1[i] + final_t[i]*emmap1.map_dim[i])*emmap1.pix[i] for i in range(3)]
                #fobj.write("   Position of the refined axis [x, y, z] (A): %s\n" %vec2string(pos_ax))
            else:
                emmap1.com1 = [emmap1.map_dim[i]//2 for i in range(3)]
                pos_ax = [(emmap1.com1[i] + final_t[i]*emmap1.map_dim[i])*emmap1.pix[i] for i in range(3)]
                #fobj.write("   Position of the refined axis [x, y, z] (A): %s\n" %vec2string(pos_ax))
        return final_axis, final_t, pos_ax
    except Exception as e:
        raise e


def axis_refine(
    emmap1,
    rotaxis,
    symorder,
    fitfsc=0.5,
    ncycles=10,
    t_init=[0.0, 0.0, 0.0],
    fobj=None,
    fitres=6,
):
    axis = np.asarray(rotaxis)
    initial_axis = axis = axis / math.sqrt(np.dot(axis, axis))
    print(" ")
    print("Initial axis and fold: ", axis, symorder)
    #if fobj is None:
    #    fobj = open("EMDA_symref.txt", "w")
    #fobj.write("Initial axis and fold: " + str(axis) + str(symorder) + "\n")
    print("Number of refinement cycles:", ncycles)
    print("Data resolution for refinement: ", fitres)
    #fobj.write("Number of refinement cycles: " + str(ncycles) + "\n")
    #fobj.write("Data resolution for refinement: " + str(fitres) + "\n")
    print("Initial axis and angles:")
    #fobj.write("Initial axis and angles: \n")

    angle = float(360.0 / symorder)
    print("   ", axis, angle)
    #fobj.write("   " + str(axis) + str(angle) + "\n")
    q = quaternions.get_quaternion(axis, angle)
    rotmat_init = quaternions.get_RM(q)
    final_axis, final_trans, axis_position = run_fit(
        emmap1=emmap1,
        rotmat=rotmat_init,
        t=np.asarray(t_init, dtype="float"),
        fitres=fitres,
        fobj=fobj,
        fitfsc=fitfsc,
        nmarchingcycles=ncycles
    )
    return initial_axis, final_axis, final_trans, axis_position


if __name__ == "__main__":
    #imap = "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/newtest/lig_full.mrc"
    #imask = "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/newtest/halfmap_mask.mrc"
    imap =  "/Users/ranganaw/MRC/REFMAC/EMD-0011/emd_0011_emda_reboxed.mrc"
    imask = "/Users/ranganaw/MRC/REFMAC/EMD-0011/emda_reboxedmask.mrc"
    #rotaxis = [0.99980629, -0.00241615,  0.01953302] # 2-fold axis
    rotaxis = [+0.000,    +0.000,    +1.000] # 3-fold axis
    symorder = 3
    ax_final, t_final = axis_refine(
            imap=imap,
            imask=imask,
            rotaxis=rotaxis, #[0.20726902, 0.97784544, 0.02928904],
            symorder=symorder
        )
    angle = float(360/symorder)
    map_output([imap], imask, ax_final, angle, t_final)
    #hf1 = "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/lig_hf1.mrc"
    #hf2 = "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/lig_hf2.mrc"
    #map_output([hf1, hf2], imask, ax_final, 180.0, t_final)