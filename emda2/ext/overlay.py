"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from emda2.core import (
    iotools,
    maptools,
    fsctools,
    restools,
    quaternions,
    plotter,
)
from emda2.ext.utils import (
    get_avg_fsc,
    determine_ibin,
    rotate_f_within_sphere,
    get_xyz_sum,
)
from scipy.ndimage.interpolation import shift
import fcodes2


class EmmapOverlay:
    def __init__(self, map_list, nocom=True, mask_list=None, modelres=None):
        self.map_list = map_list
        self.mask_list = mask_list
        self.modelres = modelres
        self.map_unit_cell = None
        self.map_origin = None
        self.map_dim = None
        self.pixsize = None
        self.arr_lst = []
        self.msk_lst = []
        self.carr_lst = []
        self.ceo_lst = None
        self.cfo_lst = None
        self.cbin_idx = None
        self.cdim = None
        self.cbin = None
        self.com = not (nocom)
        self.com1 = None
        self.comlist = []
        self.box_centr = None
        self.fhf_lst = []
        self.nbin = None
        self.res_arr = None
        self.bin_idx = None
        self.fo_lst = []
        self.eo_lst = []
        self.totalvar_lst = []
        self.gemmi = True
        self.resample_comdiff_list = []

    def _check_inputs(self):
        if not self.map_list:
            raise SystemExit("Map list is empty.")
        if len(self.map_list) < 2:
            raise SystemExit("At least 2 maps are needed.")

    def load_maps(self):
        self._check_inputs()
        for i, arr in enumerate(self.map_list):
            if self.com:
                print("Calculating COM...")
                com1 = maptools.center_of_mass_density(arr)
                if i == 0:
                    box_centr = (
                        arr.shape[0] // 2,
                        arr.shape[1] // 2,
                        arr.shape[2] // 2,
                    )
                    self.com1, self.box_centr = com1, box_centr
                self.comlist.append(com1)
                arr_mvd = shift(arr, np.subtract(box_centr, com1))
                self.arr_lst.append(arr_mvd)
                self.fhf_lst.append(fftshift(fftn(fftshift(arr_mvd))))
                outputname = "startmap_%s.mrc" % str(i)
                stmap = iotools.Map(outputname)
                stmap.arr = arr_mvd
                stmap.cell = self.map_unit_cell
                stmap.origin = self.map_origin
                stmap.write()
            else:
                self.fhf_lst.append(fftshift(fftn(fftshift(arr))))

    def calc_fsc_from_maps(self):
        # function for only two maps fitting
        nmaps = len(self.fhf_lst)
        print("nmaps: ", nmaps)
        (
            self.nbin,
            self.res_arr,
            self.bin_idx,
        ) = restools.get_resolution_array(self.map_unit_cell, self.fhf_lst[0])
        for i in range(nmaps):
            _, _, _, totalvar, fo, eo = fsctools.halfmaps_fsc_variance(
                self.fhf_lst[i], self.fhf_lst[i], self.bin_idx, self.nbin
            )
            self.fo_lst.append(fo)
            self.eo_lst.append(eo)
            self.totalvar_lst.append(totalvar)


class linefit:
    def __init__(self):
        self.e0 = None
        self.e1 = None
        self.bin_idx = None
        self.nbin = None
        self.step = None
        self.q_prev = None
        self.t = None
        self.q_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def get_linefit_static_data(self, e_list, bin_idx, res_arr, smax):
        if len(e_list) == 2:
            eout, self.bin_idx, self.nbin = restools.cut_resolution_nmaps(
                e_list, bin_idx, res_arr, smax
            )
        else:
            print("len(e_list: ", len(e_list))
            raise SystemExit()
        self.e0 = eout[0, :, :, :]
        self.e1 = eout[1, :, :, :]

    def get_fsc_wght(self, e0, ert, bin_idx, nbin):
        cx, cy, cz = e0.shape
        bin_stats = fsctools.anytwomaps_fsc_covariance(e0, ert, bin_idx, nbin)
        fsc, _ = bin_stats[0], bin_stats[1]
        # fsc = np.array(fsc, dtype=np.float64, copy=False)
        # fsc = fsc / (1 - fsc**2)
        w_grid = fcodes2.read_into_grid2(bin_idx, fsc, nbin, cx, cy, cz)[
            :, :, :, 0
        ]
        return w_grid

    def func(self, i):
        tmp = np.insert(self.step * i, 0, 0.0)
        q = tmp + self.q_prev
        q = q / np.sqrt(np.dot(q, q))
        rotmat = quaternions.get_RM(q)
        ers = rotate_f_within_sphere(rotmat, self.e1, self.bin_idx, self.nbin)
        w_grid = self.get_fsc_wght(
            self.e0, ers[:, :, :, 0], self.bin_idx, self.nbin
        )
        fval = np.real(
            np.sum(w_grid * self.e0 * np.conjugate(ers[:, :, :, 0]))
        )
        return -fval

    def scalar_opt(self, t=None):
        from scipy.optimize import minimize_scalar

        f = self.func
        res = minimize_scalar(f, method="brent")
        return res.x

    def func_t(self, i):
        nx, ny, nz = self.e0.shape
        t = self.step * i
        st, _, _, _ = fcodes2.get_st(nx, ny, nz, t)
        e1_t = self.e1 * st
        w_grid = self.get_fsc_wght(self.e0, e1_t, self.bin_idx, self.nbin)
        fval = np.sum(w_grid * self.e0 * np.conjugate(e1_t))
        return -fval.real

    def scalar_opt_trans(self):
        from scipy.optimize import minimize_scalar

        f = self.func_t
        res = minimize_scalar(f, method="brent")
        return res.x


def get_dfs(mapin, xyz, vol):
    nx, ny, nz = mapin.shape
    dfs = np.zeros(shape=(nx, ny, nz, 3), dtype=np.complex64)
    for i in range(3):
        dfs[:, :, :, i] = np.fft.fftshift(
            (1 / vol) * 2j * np.pi * np.fft.fftn(mapin * xyz[i])
        )
    return dfs


def derivatives_rotation(e0, e1, cfo, wgrid, sv, q, xyz, xyz_sum):
    # rotation derivatives
    tp2 = (2.0 * np.pi) ** 2
    nx, ny, nz = e0.shape
    vol = nx * ny * nz
    dFRS = get_dfs(np.real(ifftn(ifftshift(e1))), xyz, vol)
    dRdq = quaternions.derivatives_wrt_q(q)
    df = np.zeros(3, dtype="float")
    ddf = np.zeros((3, 3), dtype="float")
    for i in range(3):
        a = np.zeros((3, 3), dtype="float")
        for k in range(3):
            for l in range(3):
                if k == 0 or (k > 0 and l >= k):
                    a[k, l] = np.sum(
                        wgrid
                        * np.real(
                            np.conjugate(e0)
                            * (
                                dFRS[:, :, :, k]
                                * sv[l, :, :, :]
                                * dRdq[i, k, l]
                            )
                        )
                    )
                else:
                    a[k, l] = a[l, k]
        df[i] = np.sum(a)
    wfsc = wgrid * np.real(np.conjugate(e0) * e1)
    for i in range(3):
        for j in range(3):
            if i == 0 or (i > 0 and j >= i):
                b = np.zeros((3, 3), dtype="float")
                n = -1
                for k in range(3):
                    for l in range(3):
                        if k == 0 or (k > 0 and l >= k):
                            n += 1
                            b[k, l] = (
                                (-tp2 / vol)
                                * xyz_sum[n]
                                * np.sum(
                                    wfsc
                                    * sv[k, :, :, :]
                                    * sv[l, :, :, :]
                                    * dRdq[i, k, l]
                                    * dRdq[j, k, l]
                                )
                            )
                        else:
                            b[k, l] = b[l, k]
                ddf[i, j] = np.sum(b)
            else:
                ddf[i, j] = ddf[j, i]
    ddf_inv = np.linalg.pinv(ddf)
    step = ddf_inv.dot(-df)
    return step


def derivatives_translation(e0, e1, wgrid, w2grid, sv):
    PI = np.pi
    tp2 = (2.0 * PI) ** 2
    tpi = 2.0 * PI * 1j
    # translation derivatives
    df = np.zeros(3, dtype="float")
    ddf = np.zeros((3, 3), dtype="float")
    for i in range(3):
        df[i] = np.real(
            np.sum(wgrid * e0 * np.conjugate(e1 * tpi * sv[i, :, :, :]))
        )
        for j in range(3):
            if i == 0 or (i > 0 and j >= i):
                # ddf[i,j] = -tp2 * np.sum(w2grid * sv[i,:,:,:] * sv[j,:,:,:])
                ddf[i, j] = -tp2 * np.sum(
                    wgrid * sv[i, :, :, :] * sv[j, :, :, :]
                )
            else:
                ddf[i, j] = ddf[j, i]
    ddf_inv = np.linalg.pinv(ddf)
    step = ddf_inv.dot(-df)
    return step


def run_bfgs(mapobj, optmethod=None):
    from emda2.ext.bfgs import create_xyz_grid

    e0 = mapobj.ceo_lst[0]  # Static map e-data for fit
    xyz = create_xyz_grid(mapobj.cdim)
    xyz_sum = get_xyz_sum(xyz)
    e1 = mapobj.ceo_lst[1]
    cx, cy, cz = e0.shape
    t = np.array([0.0, 0.0, 0.0], dtype="float")
    st, s1, s2, s3 = fcodes2.get_st(cx, cy, cz, t)
    sv = np.array([s1, s2, s3])
    from emda2.ext import bfgs

    obj = bfgs.Bfgs()
    obj.method = optmethod
    obj.e0 = e0
    obj.e1 = e1
    obj.f1 = mapobj.cfo_lst[0]
    obj.sv = sv
    obj.xyz = xyz
    obj.xyz_sum = xyz_sum
    obj.bin_idx = mapobj.cbin_idx
    obj.nbin = mapobj.cbin
    obj.pixsize = mapobj.pixsize
    obj.map_dim = mapobj.map_dim
    obj.optimize()
    """ obj = bfgs.Bfgs_trans() 
    obj.e0 = e0
    obj.e1 = e1
    obj.sv = sv
    obj.xyz = xyz
    obj.xyz_sum = xyz_sum
    obj.bin_idx = mapobj.cbin_idx
    obj.nbin = mapobj.cbin
    obj.optimize() """
    return obj


def fsc_between_static_and_transfomed_map(maps, bin_idx, rm, t, nbin):
    nx, ny, nz = maps[0].shape
    maps2send = np.stack([maps[1], maps[2]], axis=-1)
    if not np.allclose(rm, np.eye(rm.shape[0])):
        frs = rotate_f_within_sphere(rm, maps2send, bin_idx, nbin)
    else:
        frs = maps2send
    st, _, _, _ = fcodes2.get_st(nx, ny, nz, t)
    f1f2_fsc, _, bin_count = fsctools.anytwomaps_fsc_covariance(
        maps[0], frs[:, :, :, 0] * st, bin_idx, nbin
    )
    fsc_avg = get_avg_fsc(binfsc=f1f2_fsc, bincounts=bin_count)
    return [f1f2_fsc, frs[:, :, :, 0] * st, frs[:, :, :, 1] * st, fsc_avg]


def run_fit(
    emmap1,
    rotmat,
    t,
    ncycles,
    ifit,
    fobj=None,
    fitres=None,
    optmethod="custom",
    t_only=False,
    r_only=False,
):
    if fitres is not None:
        if fitres <= emmap1.res_arr[-1]:
            fitbin = len(emmap1.res_arr) - 1
        else:
            dist = np.abs(emmap1.res_arr - fitres)
            ibin = np.argmin(dist)
            if emmap1.res_arr[ibin] < fitres:
                ibin = ibin - 1
            print("chosen resolution: ", emmap1.res_arr[ibin])
            if ibin % 2 != 0:
                ibin = ibin - 1
            fitbin = min([len(dist), ibin])
    else:
        fitbin = len(emmap1.res_arr) - 1
    fsc_lst = []
    # com difference
    comshift = np.array([0.0, 0.0, 0.0], "float")
    if len(emmap1.comlist) > 0:
        comshift = (
            np.subtract(emmap1.comlist[ifit], emmap1.comlist[0])
            * emmap1.pixsize
        )
    q = quaternions.rot2quart(rotmat)
    for i in range(10):
        print("Resolution cycle #: ", i)
        if i == 0:
            (
                f1f2_fsc,
                frt,
                ert,
                fsc_avg,
            ) = fsc_between_static_and_transfomed_map(
                maps=[
                    emmap1.fo_lst[0],
                    emmap1.fo_lst[ifit],
                    emmap1.eo_lst[ifit],
                ],
                bin_idx=emmap1.bin_idx,
                rm=rotmat,
                t=t,
                nbin=emmap1.nbin,
            )
            ibin = determine_ibin(f1f2_fsc)
            if ibin % 2 != 0:
                ibin = ibin - 1
            if fitbin < ibin:
                ibin = fitbin
            ibin_old = ibin
            print("Fitting starts at ", emmap1.res_arr[ibin], " (A)")
            fsc_lst.append(f1f2_fsc)
            if fsc_avg > 0.999:
                rotmat = rotmat
                t = t
                q_final = quaternions.rot2quart(rotmat)
                fsc_lst.append(f1f2_fsc)
                print("\n***FSC between static and moving maps***\n")
                print("bin#     resolution(A)      start-FSC     end-FSC\n")
                for j in range(len(emmap1.res_arr)):
                    print(
                        "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
                            j, emmap1.res_arr[j], fsc_lst[1][j], fsc_lst[0][j]
                        )
                    )
                break
        else:
            # Apply initial rotation and translation to calculate fsc
            f1f2_fsc, frt, ert, _ = fsc_between_static_and_transfomed_map(
                maps=[
                    emmap1.fo_lst[0],
                    emmap1.fo_lst[ifit],
                    emmap1.eo_lst[ifit],
                ],
                bin_idx=emmap1.bin_idx,
                rm=rotmat,
                t=t,
                nbin=emmap1.nbin,
            )
            ibin = determine_ibin(f1f2_fsc)
            if ibin == 0:
                print("ibin = 0, Cannot proceed! Stopping now...")
                return None
            if fitbin < ibin:
                ibin = fitbin
            if emmap1.res_arr[ibin] < 2.0:
                ibin = np.argmin(np.abs(emmap1.res_arr - 2.0))
            print("Fitting resolution: ", emmap1.res_arr[ibin], " (A)")
            if ibin_old == ibin:
                fsc_lst.append(f1f2_fsc)
                q_final = quaternions.rot2quart(rotmat)
                print("\n***FSC between static and moving maps***\n")
                print("bin#     resolution(A)      start-FSC     end-FSC\n")
                for j in range(len(emmap1.res_arr)):
                    print(
                        "{:5d} {:6.2f} {:8.4f} {:8.4f}".format(
                            j, emmap1.res_arr[j], fsc_lst[0][j], fsc_lst[1][j]
                        )
                    )
                # output fitted_map
                static_map = np.real(
                    np.fft.ifftshift(
                        np.fft.ifftn(np.fft.ifftshift(emmap1.fo_lst[0]))
                    )
                )
                fitted_map = np.real(
                    np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(frt)))
                )
                mo = iotools.Map("fitted_map.mrc")
                mo.arr = fitted_map
                mo.cell = emmap1.map_unit_cell
                mo.origin = emmap1.map_origin
                mo.write()
                ms = iotools.Map("static_map.mrc")
                ms.arr = static_map
                ms.cell = emmap1.map_unit_cell
                ms.origin = emmap1.map_origin
                ms.write()
                break
            else:
                ibin_old = ibin
        if ibin == 0:
            print("ibin = 0, Cannot proceed! Stopping now...")
            return None
        # remove negative Fs from e0 and e1
        e_list = [emmap1.eo_lst[0], ert, frt]
        eout, cBIdx, cbin = restools.cut_resolution_nmaps(
            e_list, emmap1.bin_idx, emmap1.res_arr, ibin
        )
        emmap1.ceo_lst = [eout[0, :, :, :], eout[1, :, :, :]]
        emmap1.cfo_lst = [eout[2, :, :, :]]
        emmap1.cbin_idx = cBIdx
        emmap1.cdim = eout[1, :, :, :].shape
        emmap1.cbin = cbin
        # testing pading in fourier space
        # here i cut data at low resol, but keep the grid same size
        """ nx, ny, nz = ert.shape
        eout = fcodes_fast.cutmap_arr(
            e_list, 
            emmap1.bin_idx, 
            ibin, 
            0, 
            len(emmap1.res_arr), 
            nx, ny, nz, 
            len(e_list)
            )
        emmap1.ceo_lst = [eout[0, :, :, :], eout[1, :, :, :]]
        emmap1.cfo_lst = [eout[2, :, :, :]]
        emmap1.cbin_idx = emmap1.bin_idx
        emmap1.cdim = eout[1, :, :, :].shape
        emmap1.cbin = len(emmap1.res_arr) """
        #
        if optmethod == "custom":
            from emda2.ext.custom_optimizer import Optimiser

            rfit = Optimiser(emmap1)
            rfit.comshift = comshift
            slf = ibin
            slf = min([ibin, 100])
            rfit.optimizer(
                ncycles=ncycles,
                t=t,
                rotmat=rotmat,
                ifit=ifit,
                smax_lf=slf,
                fobj=fobj,
                t_only=t_only,
                r_only=r_only,
            )
        else:
            rfit = run_bfgs(mapobj=emmap1, optmethod=optmethod)
        t += rfit.t
        q = quaternions.quaternion_multiply(rfit.q, q)
        q = q / np.sqrt(np.dot(q, q))
        rotmat = quaternions.get_RM(q)
    return [t, q_final]


def overlay(
    emmap1,
    tlist,
    qlist,
    ncycles=100,
    modelres=None,
    masklist=None,
    modellist=None,
    fitres=None,
    nocom=False,
    fobj=None,
    optmethod=None,
    r_only=False,
    t_only=False,
):
    if optmethod is None:
        optmethod = "custom"
        print("Optimization method: custom")
    if fobj is None:
        fobj = open("EMDA_overlay.txt", "+w")
    fobj.write("\n This is EMDA map overlay \n")
    rotmat_list = []
    trans_list = []
    for ifit in range(1, len(emmap1.eo_lst)):
        t, q_final = run_fit(
            emmap1=emmap1,
            rotmat=quaternions.get_RM(qlist[ifit - 1]),
            t=[
                itm / emmap1.pixsize[i]
                for i, itm in enumerate(tlist[ifit - 1])
            ],
            ncycles=ncycles,
            ifit=ifit,
            fitres=fitres,
            optmethod=optmethod,
            r_only=r_only,
            t_only=t_only,
        )
        rotmat = quaternions.get_RM(q_final)
        rotmat_list.append(rotmat)
        trans_list.append(t)
    # output maps
    # output_rotated_maps(emmap1, rotmat_list, trans_list, modellist)
    return emmap1, rotmat_list, trans_list


def output_rotated_maps(emmap1, r_lst, t_lst, modellist=[]):
    from numpy.fft import ifftn, ifftshift
    from emda2.ext.utils import shift_density

    if modellist is None:
        modellist = []
    fo_lst = emmap1.fo_lst
    cell = emmap1.map_unit_cell
    comlist = emmap1.comlist
    bin_idx = emmap1.bin_idx
    nbin = emmap1.nbin
    f_static = fo_lst[0]
    nx, ny, nz = f_static.shape
    data2write = np.real(ifftshift(ifftn(ifftshift(f_static))))
    if len(comlist) > 0:
        data2write = shift_density(
            data2write, shift=np.subtract(comlist[0], emmap1.box_centr)
        )
        f_static = fftshift(fftn(fftshift(data2write)))
    m_st = iotools.Map("static_map.mrc")
    m_st.arr = data2write
    m_st.cell = cell
    m_st.write()

    i = 0
    for fo, t, rotmat in zip(fo_lst[1:], t_lst, r_lst):
        # map output
        data2write = np.real(ifftshift(ifftn(ifftshift(fo))))
        if len(comlist) > 0:
            data2write = shift_density(
                data2write, shift=np.subtract(comlist[i + 1], emmap1.box_centr)
            )
            f_moving_before_centering = fftshift(fftn(fftshift(data2write)))
            # unaligned FSC
            fsc_before, _, _ = fsctools.anytwomaps_fsc_covariance(
                f_static, f_moving_before_centering, bin_idx, nbin
            )
        else:
            # unaligned FSC
            fsc_before, _, _ = fsctools.anytwomaps_fsc_covariance(
                f_static, fo, bin_idx, nbin
            )
        # t in pixel unit
        st, _, _, _ = fcodes2.get_st(nx, ny, nz, t)
        frt = (
            st * rotate_f_within_sphere(rotmat, fo, bin_idx, nbin)[:, :, :, 0]
        )
        data2write = np.real(ifftshift(ifftn(ifftshift(frt))))
        if len(comlist) > 0:
            data2write = shift_density(
                data2write, shift=np.subtract(comlist[0], emmap1.box_centr)
            )
            frt = fftshift(fftn(fftshift(data2write)))
        # aligned FSC
        fsc_after, _, _ = fsctools.anytwomaps_fsc_covariance(
            f_static, frt, bin_idx, nbin
        )

        # output fitted map
        m_ft = iotools.Map(
            "{0}_{1}.{2}".format("fitted_map", str(i + 1), "mrc")
        )
        m_ft.arr = data2write
        m_ft.cell = cell
        m_ft.write()

        # plot FSCs
        plotter.plot_nlines(
            emmap1.res_arr,
            [fsc_before[:nbin], fsc_after[:nbin]],
            "{0}_{1}.{2}".format("fsc", str(i + 1), "eps"),
            ["FSC before", "FSC after"],
        )
        i += 1


def output_rotated_models(emmap1, maplist, r_lst, t_lst, modellist=None):
    from emda.core.iotools import model_transform_gm

    pixsize = emmap1.pixsize
    comlist = emmap1.comlist
    resample_comdiff_list = emmap1.resample_comdiff_list
    if len(comlist) > 0:
        assert len(comlist) == len(maplist)
    i = 0
    for model, t, rotmat in zip(maplist[1:], t_lst, r_lst):
        i += 1
        t = (
            np.asarray(t, "float")
            * pixsize
            * np.asarray(emmap1.map_dim, "int")
        )
        # calculate the vector from com to box_center in Angstroms
        if len(comlist) > 0:
            shift = np.subtract(comlist[i], comlist[0]) * pixsize
            t = t + shift
        if model.endswith((".mrc", ".map")):
            continue
        else:
            outcifname = "emda_transformed_model_" + str(i) + ".cif"
            model_transform_gm(
                mmcif_file=model,
                rotmat=rotmat,
                trans=-t,
                outfilename=outcifname,
            )
    if modellist is not None:
        i = 0
        for model, t, rotmat in zip(modellist, t_lst, r_lst):
            i += 1
            t = (
                np.asarray(t, "float")
                * pixsize
                * np.asarray(emmap1.map_dim, "int")
            )
            # calculate the vector from com to box_center in Angstroms
            if len(comlist) > 0:
                shift = np.subtract(comlist[i], comlist[0]) * pixsize
                t = t + shift
                mapcom = np.asarray(comlist[i]) * pixsize
            else:
                mapcom = None
            if len(resample_comdiff_list) > 0:
                t += resample_comdiff_list[i - 1]
            outcifname = "emda_transformed_modellist_" + str(i) + ".cif"
            model_transform_gm(
                mmcif_file=model,
                rotmat=rotmat,
                trans=-t,
                mapcom=mapcom,
                outfilename=outcifname,
            )


if __name__ == "__main__":
    maplist = [
        # "/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3651/emda_test/map_transform/emd_3651.map",
        # "/Users/ranganaw/MRC/REFMAC/haemoglobin/EMD-3651/emda_test/map_transform/transformed.mrc"
        # "/Users/ranganaw/MRC/REFMAC/EMD-6952/emda_test/map_transform/emd_6952.map",
        # "/Users/ranganaw/MRC/REFMAC/EMD-6952/emda_test/map_transform/transformed.mrc",
        # "/Users/ranganaw/MRC/REFMAC/Vinoth/emda_test/diffmap/postprocess_nat.mrc",
        # "/Users/ranganaw/MRC/REFMAC/Vinoth/emda_test/diffmap/postprocess_lig.mrc",
        # "/Users/ranganaw/MRC/REFMAC/Takanori_ATPase/EMD-9931/emd_9931.map",
        # "/Users/ranganaw/MRC/REFMAC/Takanori_ATPase/EMD-9934/emd_9934.map",
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/emd_21997.map",
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/emd_21999.map",
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/extracted_rbds/proshade_fit/ProShade_1/rotStr.map"
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/extracted_rbds/21997_RDB.mrc",
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/extracted_rbds/21999_RDB.mrc"
        # "/Users/ranganaw/MRC/REFMAC/crop_refmactools_01/outward/postprocess_masked_cut.mrc",
        # "/Users/ranganaw/MRC/REFMAC/crop_refmactools_01/melphalan/postprocess_masked_cut.mrc"
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/test3/fitted_map_1.mrc",
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/test3/6x2a.cif"
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/test3/21999_coordinate_fit/closed_rbd_fit/21997.mrc",
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/test3/21999_coordinate_fit/closed_rbd_fit/21999_fitted_on_21997.mrc"
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/test3/21999_coordinate_fit/closed_rbd_fit/coordinate_fit/21999_closed_RBD_fitted.mrc",
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/test3/21999_coordinate_fit/closed_rbd_fit/coordinate_fit/6x2a_closed_RBD.pdb",
        "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/newtest/lig_full.mrc",
        "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/newtest/ligfull_rotated_2fold_refined.mrc",
    ]

    masklist = [
        # "/Users/ranganaw/MRC/REFMAC/Takanori_ATPase/DomainMasks/6k7g-9931_A_mask.mrc",
        # "/Users/ranganaw/MRC/REFMAC/Takanori_ATPase/DomainMasks/6k7i-9934_A_mask.mrc"
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/rdb_fit/emda_atomic_mask_6x29_RDB.mrc",
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/rdb_fit/emda_atomic_mask_6x2a_RDB.mrc"
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/test3/21999_coordinate_fit/closed_rbd_fit/emda_atomic_mask_6x29_closed_RBD.mrc",
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/test3/21999_coordinate_fit/closed_rbd_fit/emda_atomic_mask_6x2a_closed_RBD.mrc"
    ]

    modellist = [
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997//reworked_RBDmasks/6x2a_transformed_RDB_trimmed.pdb"
        # "/Users/ranganaw/MRC/REFMAC/COVID19/EMD-21997/6x2a.cif"
    ]
    # emmap1, rotmat_lst, transl_lst = overlay(maplist=maplist, masklist=masklist)
    # emmap1, rotmat_lst, transl_lst = overlay(maplist=maplist, modellist=modellist)
    emmap1, rotmat_lst, transl_lst = overlay(maplist=maplist, modelres=3.5)
