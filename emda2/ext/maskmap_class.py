"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

# this code generates the mask if not supplied.
# then ally that mask to arr.
from __future__ import absolute_import, division, print_function, unicode_literals
import emda2.core as core
import numpy as np


class MaskedMaps:
    def __init__(self, hfmap_list=None):
        self.hfmap_list = hfmap_list
        self.mask = None
        self.uc = None
        self.arr1 = None
        self.arr2 = None
        self.origin = None
        self.iter = 3
        self.smax = 9 # kernel radius in pixels
        self.prob = 0.99
        self.dthresh = None

    def read_halfmaps(self):
        map1 = core.iotools.Map(self.hfmap_list[0])
        map1.read()
        map2 = core.iotools.Map(self.hfmap_list[1])
        map2.read()
        self.uc = map1.cell
        self.arr1 = map1.arr
        self.arr2 = map2.arr
        self.origin = map1.origin

    def generate_mask(self):
        from emda.ext import realsp_local
        kern = core.restools.create_soft_edged_kernel_pxl(self.smax)
        self.arr1 = threshold_map(arr=self.arr1, prob=self.prob, dthresh=self.dthresh)
        self.arr2 = threshold_map(arr=self.arr2, prob=self.prob, dthresh=self.dthresh)
        halfcc3d = realsp_local.get_3d_realspcorrelation(self.arr1, self.arr2, kern)
        self.mask = self.thereshol_ccmap(ccmap=halfcc3d)

    def create_edgemask(self, radius):
        # Remove everything outside radius
        box_radius = radius + 1
        box_size = radius * 2 + 1
        # Creating a sphere mask
        center = [box_radius, box_radius, box_radius]
        print("boxsize: ", box_size, "boxradius: ", box_radius, "center:", center)
        radius = box_radius
        X, Y, Z = np.ogrid[:box_size, :box_size, :box_size]
        dist_from_center = np.sqrt(
            (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        )
        mask = dist_from_center <= radius
        return mask

    def histogram2(self, arr, prob=0.99):
        from scipy import stats
        from scipy.ndimage.morphology import binary_dilation
        import matplotlib
        matplotlib.use(matplotlib.get_backend())
        import matplotlib.pyplot as plt

        arr_tmp = arr
        X2 = np.sort(arr.flatten())
        F2 = np.array(range(len(X2))) / float(len(X2) - 1)
        loc = np.where(F2 >= prob)
        thresh = X2[loc[0][0]]
        #thresh = max([thresh, np.max(X2) * 0.02])
        #print("threshold: ", thresh) 
        # plot the sorted data:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.plot(F2, X2)
        ax1.set_xlabel("$p$")
        ax1.set_ylabel("$x$")
        ax2 = fig.add_subplot(122)
        ax2.plot(X2, F2)
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$p$")
        plt.savefig("cdf.png", format="png", dpi=300)
        nx, ny, nz = arr.shape
        maxbin = np.amax(np.array([nx // 2, ny // 2, nz // 2]))
        counts = stats.binned_statistic(
            arr.flatten(), arr.flatten(), statistic="count", bins=maxbin, range=(0, 1)
        )[0]
        sc = counts - np.roll(counts, 1)
        ulim = len(sc) - 11 + np.argmax(sc[-10:])
        edge_mask = self.create_edgemask(ulim)
        cc_mask = np.zeros(shape=(nx, ny, nz), dtype="bool")
        cx, cy, cz = edge_mask.shape
        dx = (nx - cx) // 2
        dy = (ny - cy) // 2
        dz = (nz - cz) // 2
        print(dx, dy, dz)
        cc_mask[dx : dx + cx, dy : dy + cy, dz : dz + cz] = edge_mask
        arr_tmp = arr_tmp * cc_mask
        binary_arr = (arr_tmp > thresh).astype(int)
        dilate = binary_dilation(binary_arr, iterations=self.iter)
        mask = make_soft(dilate, kern_rad=self.smax)
        mask = mask * (mask >= 1e-3)
        return mask

    def thereshol_ccmap(self, ccmap):
        from scipy.ndimage.morphology import binary_dilation

        ccmap_binary = (ccmap >= 1e-3).astype(int)
        dilate = binary_dilation(ccmap_binary, iterations=self.iter)
        mask = make_soft(dilate, kern_rad=2)
        return mask * (mask >= 1e-3)      

    def histogram(self, arr1):
        from scipy import stats

        nx, ny, nz = arr1.shape
        maxbin = np.amax(np.array([nx // 2, ny // 2, nz // 2]))
        counts = stats.binned_statistic(
            arr1.flatten(), arr1.flatten(), statistic="count", bins=maxbin, range=(0, 1)
        )[0]
        sc = counts - np.roll(counts, 1)
        ulim = len(sc) - 11 + np.argmax(sc[-10:])
        cc_arr = np.linspace(0, 1, maxbin)
        xc = cc_arr[:ulim] * counts[:ulim]
        xc_sum = np.sum(xc)
        isum = 0.0
        cc_arr = cc_arr[:ulim]
        for i in range(len(xc)):
            isum = isum + xc[i]
            if isum >= xc_sum / 2:
                threshold = cc_arr[i]
                break
        edge_mask = self.create_edgemask(ulim)
        cc_mask = np.zeros(shape=(nx, ny, nz), dtype="bool")
        cx, cy, cz = edge_mask.shape
        dx = (nx - cx) // 2
        dy = (ny - cy) // 2
        dz = (nz - cz) // 2
        print(dx, dy, dz)
        cc_mask[dx : dx + cx, dy : dy + cy, dz : dz + cz] = edge_mask
        return cc_mask, threshold

    def get_radial_sum(self, arr1):
        # this function not used
        from matplotlib import pyplot as plt

        nx, ny, nz = arr1.shape
        nbin, res_arr, bin_idx, _ = core.restools.get_resolution_array(self.uc, arr1)
        sum_lst = []
        ibin_lst = []
        isum = 0.0
        isum_old = 0.0
        for ibin in range(nbin):
            ibin_sum = np.sum(arr1 * (bin_idx == ibin))
            ibin_lst.append(ibin_sum)
            isum_old = isum
            isum = isum + ibin_sum
            # if ibin_sum < 0.0 and ibin > 10:
            if isum <= isum_old and ibin > 10:
                break
            print(ibin, ibin_sum)
            sum_lst.append(isum)
        edge_mask = self.create_edgemask(ibin)
        cc_mask = np.zeros(shape=(nx, ny, nz), dtype="bool")
        cx, cy, cz = edge_mask.shape
        dx = (nx - cx) // 2
        dy = (ny - cy) // 2
        dz = (nz - cz) // 2
        print(dx, dy, dz)
        cc_mask[dx : dx + cx, dy : dy + cy, dz : dz + cz] = edge_mask
        core.iotools.write_mrc(arr1 * cc_mask, "maskedcc_map.mrc", self.uc, self.origin)
        plt.plot(ibin_lst, "r")
        plt.show()


def threshold_map(arr, prob = 0.99, dthresh=None):
    if dthresh is None:
        X2 = np.sort(arr.flatten())
        F2 = np.array(range(len(X2))) / float(len(X2) - 1)
        loc = np.where(F2 >= prob)
        thresh = X2[loc[0][0]]
    else:
        thresh = dthresh
    return arr * (arr > thresh)


def binary_dilation_ccmask(ccmask, iter=1):
    from scipy.ndimage.morphology import binary_dilation

    return binary_dilation(ccmask, iterations=iter).astype(ccmask.dtype)


def make_soft(dilated_mask, kern_rad=3):
    # convoluting with gaussian sphere
    import scipy.signal

    kern_sphere = core.restools.create_soft_edged_kernel_pxl(kern_rad)
    return scipy.signal.fftconvolve(dilated_mask, kern_sphere, "same")


def mapmask(arr, uc, itr=3, kern_rad=3, prob=0.99):
    from scipy.ndimage.morphology import binary_dilation
    
    arr_tmp = threshold_map(arr=arr, prob=prob)
    binary_arr = (arr_tmp > 1e-5).astype(int)
    dilate = binary_dilation(binary_arr, iterations=itr)
    mask = make_soft(dilate, kern_rad)
    mask = mask * (mask >= 0.0)
    return np.where(binary_arr, 1.0, mask)


def mask_from_coordinates(mapname, modelname, atmrad=3, binary_mask=False):
    import scipy.signal
    from scipy.ndimage.morphology import binary_dilation
    from emda2.core import iotools, modeltools

    mapobj1 = iotools.Map(name=mapname)
    mapobj1.read()
    arr = mapobj1.arr
    uc = mapobj1.cell
    orig = mapobj1.origin
    grid_3d = np.zeros((arr.shape), dtype='float')
    if modelname.endswith((".pdb", ".ent")):
        modeltools.pdb2mmcif(modelname)
        _, x_np, y_np, z_np, _ = iotools.read_mmcif('out.cif')
    elif modelname.endswith((".cif")):
        _, x_np, y_np, z_np, _ = iotools.read_mmcif(modelname)
    # now map model coords into the 3d grid. (approximate grid positions)
    x = (x_np * arr.shape[2] / uc[2])
    y = (y_np * arr.shape[1] / uc[1])
    z = (z_np * arr.shape[0] / uc[0])
    for ix, iy, iz in zip(x, y, z):
        grid_3d[int(round(ix)), 
                int(round(iy)),
                int(round(iz))] = 1.0
    # now convolute with sphere
    pixsize = uc[0] / arr.shape[0]
    kern_rad = int(atmrad / pixsize) + 1
    print("kernel radius: ", kern_rad)
    grid2 = scipy.signal.fftconvolve(
        grid_3d, core.restools.create_binary_kernel(kern_rad), "same")
    grid2_binary = grid2 > 1e-5
    # dilate
    dilate = binary_dilation(grid2_binary, iterations=1)
    # smoothening
    mask = scipy.signal.fftconvolve(
        dilate, core.restools.softedgekernel_5x5(), "same")
    mask = mask * (mask >= 1.e-5)
    mask = np.where(grid2_binary, 1.0, mask)
    #shift_z = mask.shape[0] - abs(orig[2])
    #shift_y = mask.shape[1] - abs(orig[1])
    #shift_x = mask.shape[2] - abs(orig[0])
    shift_z = (orig[0])
    shift_y = (orig[1])
    shift_x = (orig[2])
    mask = np.roll(
        np.roll(np.roll(mask, -shift_z, axis=0), -shift_y, axis=1),
        -shift_x,
        axis=2,
    )
    """ core.iotools.write_mrc(mask, "emda_atomic_mask.mrc", uc, orig)
    if binary_mask:      
        core.iotools.write_mrc(mask > 1e-4, "emda_atomic_mask_binary.mrc", uc, orig) """
    mapobj1.arr = mask #np.transpose(mask)
    return mapobj1


if __name__ == "__main__":
    """ maplist = [
        "/Users/ranganaw/MRC/REFMAC/Bianka/EMD-4572/other/run_half1_class001_unfil.mrc",
        "/Users/ranganaw/MRC/REFMAC/Bianka/EMD-4572/other/run_half2_class001_unfil.mrc",
    ]
    obj = MaskedMaps(maplist)
    obj.read_halfmaps()
    obj.generate_mask(obj.arr1, obj.arr2) """
