# EMDA custom optimizer
import numpy as np
import emda2.emda_methods2 as em
import fcodes2

# from emda2.core import fsctools
from emda2.ext.overlay import get_avg_fsc
from emda2.ext.utils import set_array
from emda2.core import quaternions
from timeit import default_timer as timer
import fcodes2
from emda2.ext.utils import (
    get_avg_fsc,
    # determine_ibin,
    # rotate_f_within_sphere,
    get_xyz_sum,
    cut_resolution_for_linefit,
)
from emda2.ext.bfgs import create_xyz_grid
from emda2.ext.utils import cut_resolution_for_linefit
from scipy.optimize import minimize_scalar

# from emda2.ext.bfgs import get_rgi, get_f

timeit = False
debug_mode = 0


def calc_fsc(f1, f2, bin_idx, nbin):
    nx, ny, nz = f1.shape
    (
        f1f2_covar,
        binfsc,
        bincounts,
    ) = fcodes2.calc_covar_and_fsc_betwn_anytwomaps(
        f1, f2, bin_idx, nbin, debug_mode, nx, ny, nz
    )
    fsc_avg = get_avg_fsc(binfsc=binfsc, bincounts=bincounts)
    return [binfsc, fsc_avg]


def get_dfs(mapin, xyz, vol):
    nx, ny, nz = mapin.shape
    dfs = np.zeros(shape=(nx, ny, nz, 3), dtype=np.complex64)
    for i in range(3):
        dfs[:, :, :, i] = np.fft.fftshift(
            (1 / vol) * 2j * np.pi * np.fft.fftn(mapin * xyz[i])
        )
    return dfs


def derivatives_rotation(e0, e1, wgrid, sv, q, xyz, xyz_sum):
    # rotation derivatives
    tp2 = (2.0 * np.pi) ** 2
    nx, ny, nz = e0.shape
    vol = nx * ny * nz
    dFRS = get_dfs(np.real(np.fft.ifftn(np.fft.ifftshift(e1))), xyz, vol)
    # dFRS = get_dfs2(e1, xyz)
    # temp = np.gradient(e1) # numerical derivative
    # dFRS = np.zeros((nx, ny, nz, 3), 'complex')
    # for i in range(3):
    #    dFRS[:,:,:,i] = temp[i]
    # dRdq = derivatives_wrt_q(q)
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
    start = timer()
    # translation derivatives
    df = np.zeros(3, dtype="float")
    ddf = np.zeros((3, 3), dtype="float")
    for i in range(3):
        df[i] = np.real(
            np.sum(wgrid * e0 * np.conjugate(e1 * tpi * sv[i, :, :, :]))
        )
        for j in range(3):
            if i == 0 or (i > 0 and j >= i):
                ddf[i, j] = -tp2 * np.sum(
                    wgrid * sv[i, :, :, :] * sv[j, :, :, :]
                )
            else:
                ddf[i, j] = ddf[j, i]
    ddf_inv = np.linalg.pinv(ddf)
    step = ddf_inv.dot(-df)
    end = timer()
    # print("time for trans deriv. ", end-start)
    return step


class Optimiser:
    def __init__(self, mapobj, interp="linear", dfs=None):
        self.mapobj = mapobj
        self.cut_dim = mapobj.cdim
        self.ful_dim = mapobj.map_dim
        self.cell = mapobj.map_unit_cell
        self.pixsize = mapobj.pixsize
        self.origin = mapobj.map_origin
        self.interp = interp
        self.dfs = dfs
        self.w_grid = None
        self.fsc = None
        self.sv = None
        self.t = None
        self.st = None
        self.step = None
        self.q = None
        self.q_accum = None
        self.q_final_list = []
        self.rotmat = None
        self.t_accum = None
        self.ert = None
        self.frt = None
        self.cfo = None
        self.crt = None
        self.e0 = None
        self.e1 = None
        self.w2_grid = None
        self.fsc_lst = []
        self.le0 = None
        self.le1 = None
        self.lbinindx = None
        self.lnbin = None
        self.finalrun = False  # test option
        self.comshift = None

    def calc_fsc(self):
        nx, ny, nz = self.e0.shape
        (
            f1f2_covar,
            binfsc,
            bincounts,
        ) = fcodes2.calc_covar_and_fsc_betwn_anytwomaps(
            self.e0,
            self.ert,
            self.mapobj.cbin_idx,
            self.mapobj.cbin,
            debug_mode,
            nx,
            ny,
            nz,
        )
        # return average fsc as well
        from emda2.ext.overlay import get_avg_fsc

        fsc_avg = get_avg_fsc(binfsc=binfsc, bincounts=bincounts)
        return [binfsc, fsc_avg]

    def get_wght(self):
        cx, cy, cz = self.e0.shape
        val_arr = np.zeros((self.mapobj.cbin, 2), dtype="float")
        binfsc = set_array(arr=self.fsc, thresh=0.0)
        w1 = binfsc / (1 - binfsc**2)
        w2 = w1**2
        val_arr[:, 0] = self.fsc  # / (1 - self.fsc ** 2)
        fsc_sqd = self.fsc**2
        fsc_combi = fsc_sqd  # / (1 - fsc_sqd)
        val_arr[:, 1] = fsc_combi
        # val_arr[:,0] = w1
        # val_arr[:,1] = w2
        wgrid = fcodes2.read_into_grid2(
            self.mapobj.cbin_idx, val_arr, self.mapobj.cbin, cx, cy, cz
        )
        return wgrid[:, :, :, 0], wgrid[:, :, :, 1]

    def functional(self):
        fval = np.sum(self.w_grid * self.e0 * np.conjugate(self.ert))
        return fval.real

    def optimizer(
        self,
        ncycles,
        t,
        rotmat,
        ifit,
        smax_lf,
        t_only=False,
        r_only=False,
        fobj=None,
        q_init=None,
    ):
        rotation = translation = True
        if t_only:
            rotation = False
        if r_only:
            translation = False
        tol = 1e-2
        fsc_lst = []
        fval_list = []
        self.e0 = self.mapobj.ceo_lst[0]  # Static map e-data for fit
        xyz = create_xyz_grid(self.cut_dim)
        xyz_sum = get_xyz_sum(xyz)
        #t_init = np.asarray(t)
        if q_init is None:
            q_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            q_init = np.asarray(q_init)
        print("Cycle#   ", "Fval  ", "Rot(deg)  ", "Trans(A)  ", "avg(FSC)")
        q0 = quaternions.rot2quart(rotmat)
        self.e1 = self.mapobj.ceo_lst[1]
        self.cfo = self.mapobj.cfo_lst[0]
        assert np.ndim(rotmat) == 2
        for i in range(ncycles):
            start = timer()
            cx, cy, cz = self.e0.shape
            if i == 0:
                self.t = np.array([0.0, 0.0, 0.0], dtype="float")
                self.st, s1, s2, s3 = fcodes2.get_st(cx, cy, cz, self.t)
                self.sv = np.array([s1, s2, s3])
                self.q = q_init
                self.rotmat = quaternions.get_RM(self.q)
                self.ert = self.e1
                self.crt = self.cfo
            else:
                if rotation and not translation:
                    # rotation only
                    self.rotmat = quaternions.get_RM(self.q)
                    maps2send = np.stack((self.e1, self.cfo), axis=-1)
                    bin_idx = self.mapobj.cbin_idx
                    nbin = self.mapobj.cbin
                    # maps = fcodes2.trilinear2(maps2send,bin_idx,self.rotmat,nbin,0,2,cx, cy, cz)
                    maps = fcodes2.trilinear_sphere(
                        self.rotmat, maps2send, bin_idx, 0, nbin, 2, cx, cy, cz
                    )
                    self.ert = maps[:, :, :, 0]
                    self.crt = maps[:, :, :, 1]
                elif translation and not rotation:
                    # translation only
                    self.st, s1, s2, s3 = fcodes2.get_st(cx, cy, cz, self.t)
                    self.sv = np.array([s1, s2, s3])
                    self.ert = self.e1 * self.st
                    self.crt = self.cfo * self.st
                else:
                    # both rotation and translation need to refine
                    # first rotate
                    self.rotmat = quaternions.get_RM(self.q)
                    maps2send = np.stack((self.e1, self.cfo), axis=-1)
                    bin_idx = self.mapobj.cbin_idx
                    nbin = self.mapobj.cbin
                    # maps = fcodes2.trilinear2(maps2send,bin_idx,self.rotmat,nbin,0,2,cx, cy, cz)
                    maps = fcodes2.trilinear_sphere(
                        self.rotmat, maps2send, bin_idx, 0, nbin, 2, cx, cy, cz
                    )
                    # then translate
                    self.st, s1, s2, s3 = fcodes2.get_st(cx, cy, cz, self.t)
                    self.sv = np.array([s1, s2, s3])
                    self.ert = maps[:, :, :, 0] * self.st
                    self.crt = maps[:, :, :, 1] * self.st
            self.fsc, fsc_avg = self.calc_fsc()
            fsc_lst.append(self.fsc)
            self.w_grid, self.w2_grid = self.get_wght()
            fval = self.functional()
            fval_list.append(fval)
            # current rotation and translation to print
            q = quaternions.quaternion_multiply(self.q, q0)
            q = q / np.sqrt(np.dot(q, q))
            theta2 = (
                np.arccos((np.trace(quaternions.get_RM(q)) - 1) / 2)
                * 180.0
                / np.pi
            )
            t_accum_angstrom = (
                (t + self.t) * self.pixsize * self.mapobj.map_dim[0]
            )
            if self.comshift is not None:
                t_accum_angstrom += self.comshift
            translation_vec = np.sqrt(
                np.dot(t_accum_angstrom, t_accum_angstrom)
            )
            # check for convergence
            if rotation and not translation:
                if i > 2:
                    if (
                        abs(fval_list[-3] - fval_list[-2]) < tol
                        and abs(fval_list[-2] - fval_list[-1]) < tol
                    ):
                        break
            elif translation and not rotation:
                if i > 2:
                    if (
                        abs(fval_list[-3] - fval_list[-2]) < tol
                        and abs(fval_list[-2] - fval_list[-1]) < tol
                    ):
                        break
            else:
                if i > 5:
                    if (
                        abs(fval_list[-3] - fval_list[-2]) < tol
                        and abs(fval_list[-2] - fval_list[-1]) < tol
                    ):
                        break
            if rotation and not translation:
                self.step = derivatives_rotation(
                    self.e0,
                    self.ert,
                    self.w_grid,
                    self.sv,
                    self.q,
                    xyz,
                    xyz_sum,
                )
                lft = lft = linefit()  # ndlinefit()
                lft.get_linefit_static_data(
                    [self.e0, self.ert],
                    self.mapobj.cbin_idx,
                    self.mapobj.res_arr,
                    smax_lf,
                )
                lft.step = self.step
                alpha_r = lft.scalar_opt()
                self.q += np.insert(self.step * alpha_r, 0, 0.0)
                self.q = self.q / np.sqrt(np.dot(self.q, self.q))
            elif translation and not rotation:
                # translation optimisation
                self.step = derivatives_translation(
                    self.e0, self.ert, self.w_grid, self.w2_grid, self.sv
                )
                lft = linefit()
                lft.get_linefit_static_data(
                    [self.e0, self.ert],
                    self.mapobj.cbin_idx,
                    self.mapobj.res_arr,
                    smax_lf,
                )
                lft.step = self.step
                alpha_t = lft.scalar_opt_trans()
                self.t += self.step * alpha_t
            else:
                if i % 2 == 0:
                    self.step = derivatives_rotation(
                        self.e0,
                        self.ert,
                        self.w_grid,
                        self.sv,
                        self.q,
                        xyz,
                        xyz_sum,
                    )
                    lft = linefit()  # ndlinefit()
                    lft.get_linefit_static_data(
                        [self.e0, self.ert],
                        self.mapobj.cbin_idx,
                        self.mapobj.res_arr,
                        smax_lf,
                    )
                    lft.step = self.step
                    alpha_r = lft.scalar_opt()
                    print("q-step, alpha_r ", self.step, alpha_r)
                    self.q += np.insert(self.step * alpha_r, 0, 0.0)
                    self.q = self.q / np.sqrt(np.dot(self.q, self.q))
                else:
                    # translation optimisation
                    self.step = derivatives_translation(
                        self.e0, self.ert, self.w_grid, self.w2_grid, self.sv
                    )
                    lft = linefit()
                    lft.get_linefit_static_data(
                        [self.e0, self.ert],
                        self.mapobj.cbin_idx,
                        self.mapobj.res_arr,
                        smax_lf,
                    )
                    lft.step = self.step
                    alpha_t = lft.scalar_opt_trans()
                    print("t-step, alpha_t ", self.step, alpha_t)
                    self.t += self.step * alpha_t
            print(
                "{:5d} {:8.4f} {:6.4f} {:6.4f} {:8.7f}".format(
                    i, fval, theta2, translation_vec, fsc_avg
                )
            )
            end = timer()
            if timeit:
                print("time for one cycle:", end - start)


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
            eout, self.bin_idx, self.nbin = cut_resolution_for_linefit(
                e_list, bin_idx, res_arr, smax
            )
        else:
            print("len(e_list): ", len(e_list))
            raise SystemExit()
        self.e0 = eout[0, :, :, :]
        self.e1 = eout[1, :, :, :]

    def get_fsc_wght(self, e0, ert, bin_idx, nbin):
        cx, cy, cz = e0.shape
        # print('TRANSLATION')
        fsc = calc_fsc(e0, ert, bin_idx, nbin)[0]
        w_grid = fcodes2.read_into_grid(bin_idx, fsc, nbin, cx, cy, cz)
        return w_grid

    def func(self, i):
        nx, ny, nz = self.e0.shape
        tmp = np.insert(self.step * i, 0, 0.0)
        q = tmp + self.q_init  # self.q_prev
        q = q / np.sqrt(np.dot(q, q))
        rotmat = quaternions.get_RM(q)
        ers = get_FRS(rotmat, self.e1, interp="linear")
        w_grid = self.get_fsc_wght(
            self.e0, ers[:, :, :, 0], self.bin_idx, self.nbin
        )
        fval = np.real(
            np.sum(w_grid * self.e0 * np.conjugate(ers[:, :, :, 0]))
        )
        # ers = get_f(self.e1, self.ereal_rgi, self.eimag_rgi, rotmat)
        # w_grid = self.get_fsc_wght(self.e0, ers, self.bin_idx, self.nbin)
        # fval = np.real(np.sum(w_grid * self.e0 * np.conjugate(ers)))
        return -fval / (0.5 * nx * ny * nz)

    def scalar_opt(self, t=None):
        # self.ereal_rgi, self.eimag_rgi = get_rgi(self.e1)
        f = self.func
        res = minimize_scalar(f, method="brent", tol=1e-5)
        return res.x

    def func_t(self, i):
        nx, ny, nz = self.e0.shape
        t = self.step * i
        st, _, _, _ = fcodes2.get_st(nx, ny, nz, t)
        e1_t = self.e1 * st
        w_grid = self.get_fsc_wght(self.e0, e1_t, self.bin_idx, self.nbin)
        fval = np.sum(w_grid * self.e0 * np.conjugate(e1_t))
        return -fval.real / (0.5 * nx * ny * nz)

    def scalar_opt_trans(self):
        start = timer()
        f = self.func_t
        res = minimize_scalar(f, method="brent", tol=1e-5)
        end = timer()
        return res.x


class ndlinefit:
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
            eout, self.bin_idx, self.nbin = cut_resolution_for_linefit(
                e_list, bin_idx, res_arr, smax
            )
        else:
            print("len(e_list: ", len(e_list))
            raise SystemExit()
        self.e0 = eout[0, :, :, :]
        self.e1 = eout[1, :, :, :]

    def get_fsc_wght(self, e0, ert, bin_idx, nbin):
        cx, cy, cz = e0.shape
        """ bin_stats = core.fsc.anytwomaps_fsc_covariance(e0, ert, bin_idx, nbin)
        fsc, _ = bin_stats[0], bin_stats[1] """
        # print('ROTATION')
        fsc = calc_fsc(e0, ert, bin_idx, nbin)[0]
        w_grid = fcodes2.read_into_grid(bin_idx, fsc, nbin, cx, cy, cz)
        return w_grid

    def func(self, init_guess):
        from emda.ext.mapfit import utils as maputils

        tmp = np.insert(
            np.asarray(self.step, "float") * np.asarray(init_guess, "float"),
            0,
            0.0,
        )
        # tmp = np.insert(self.step * i, 0, 0.0)
        # q = tmp + self.q_prev
        q = tmp + self.q_init
        q = q / np.sqrt(np.dot(q, q))
        rotmat = quaternions.get_RM(q)
        ers = maputils.get_FRS(rotmat, self.e1, interp="linear")
        # w_grid = self.get_fsc_wght(self.e0, ers[:, :, :, 0], self.bin_idx, self.nbin)
        # fval = np.real(np.sum(w_grid * self.e0 * np.conjugate(ers[:, :, :, 0])))
        fval = np.real(np.sum(self.e0 * np.conjugate(ers[:, :, :, 0])))
        return -fval

    def scalar_opt(self, t=None):
        import scipy.optimize as optimize

        f = self.func
        init_guess = np.array([1.0, 1.0, 1.0], dtype="float")
        res = optimize.minimize(f, init_guess, method="Powell")
        return np.asarray(res.x, "float")
