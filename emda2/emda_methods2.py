""" 
This is EMDA2 API
Methods in this script encapsulates functions
implemented in other modules. 
Copyright - R. Warshamanage and G. N. Murshudov
"""

from emda2.core import iotools, maptools, restools, fsctools
import fcodes2
import numpy as np
import gemmi, shutil, os
from numpy.fft import fftshift, fftn

debug_mode = 0


def get_binidx(cell, arr):
    nbin, res_arr, bin_idx, sgrid = restools.get_resolution_array(
        uc=cell, hf1=arr
    )
    return nbin, res_arr, bin_idx, sgrid


def get_map_power(fo, bin_idx, nbin):
    """Calculates the map power spectrum.

    Arguments:
        Inputs:
            fo: complex Fourier coefficeints, ndarray
            bin_idx: 3D grid of bin numbers, integer
            nbin: number of resolution bins, integer

        Outputs:
            power_spectrum: float, 1D array
                Rotationally averaged power spectrum.
    """
    power_spectrum = maptools.get_map_power(fo=fo, bin_idx=bin_idx, nbin=nbin)
    return power_spectrum


def get_normalized_sf(fo, bin_idx, nbin):
    """Calculates normalised Fourier coefficients.
    Fourier coefficients are normalised by their radial
    power in bins.

    Arguments:
        Inputs:
            fo: complex Fourier coefficeints, ndarray
            bin_idx: 3D grid of bin numbers, integer
            nbin: number of resolution bins, integer

        Outputs:
            eo: complex, normalized Fourier coefficients.
    """
    nx, ny, nz = fo.shape
    eo = fcodes2.get_normalized_sf_singlemap(
        fo=fo,
        bin_idx=bin_idx,
        nbin=nbin,
        mode=0,
        nx=nx,
        ny=ny,
        nz=nz,
    )
    return eo


def fsc(f1, f2, bin_idx, nbin, fobj=None, xmlobj=None):
    """Returns Fourier Shell Correlation (FSC) between any two maps.

    Computes Fourier Shell Correlation (FSC) using any two maps.

    Arguments:
        Inputs:
            f1, f2: complex Fourier coefficients, ndarray
                Corresponding to map1 and map2
            bin_idx: 3D grid of bin numbers, integer
            nbin: number of resolution bins, integer

        Outputs:
            bin_fsc: float, 1D array
                FSC in each resolution bin.
    """

    bin_fsc = fsctools.anytwomaps_fsc_covariance(f1, f2, bin_idx, nbin)[0]
    return bin_fsc


def halfmap_fsc(f_hf1, f_hf2, bin_idx, nbin, filename=None):
    """Computes Fourier Shell Correlation (FSC) using half maps.

    Computes Fourier Shell Correlation (FSC) using half maps.
    FSC is not corrected for mask effect in this implementation.

    Arguments:
        Inputs:
            f_hf1, f_hf2: complex Fourier coefficients of
                halfmap1 and halfmap2. ndarrays
            bin_idx: 3D grid of bin numbers, integer
            nbin: number of resolution bins, integer

        Outputs:
            bin_fsc: float, 1D array
                Linear array of FSC in each resolution bin.
    """
    (
        _,
        _,
        noisevar,
        signalvar,
        totalvar,
        bin_fsc,
        bincount,
    ) = fcodes2.calc_fsc_using_halfmaps(
        f_hf1,
        f_hf2,
        bin_idx,
        nbin,
        debug_mode,
        f_hf1.shape[0],
        f_hf1.shape[1],
        f_hf1.shape[2],
    )
    return bin_fsc


def mask_from_map(
    uc,
    arr,
    kern=5,
    resol=15,
    filter="butterworth",
    order=1,
    prob=0.99,
    itr=3,
):
    """Generates a mask from a map.

    Generates a mask from a map.

    Arguments:
        Inputs:
            uc: float, 1D array
                Unit cell parameters.
            arr: float, 3D array
                Map data.
            half2: float, 3D array
                Half map 2 data.
            kern: integer, optional
                Radius of integrating kernel in voxels. Default is 5.
            resol: float, optional
                Resolution cutoff for lowpass filtering in Angstrom units.
                Default is 15 Angstrom.
            filter: string,optional
                Filter type to use with lowpass filtering. Default is butterworth.
            order: integer, optional
                Butterworth filter order. Default is 1.
            prob: float, optional
                Cumulative probability cutoff to decide the density threshold.
                Default value is 0.99.
            itr: integer, optional
                Number of dilation cycles. Default is 3 cycles.

        Outputs:
            mask: float, 3D array
                3D Numpy array of the mask.
    """
    from emda2.ext import maskmap_class

    _, arrlp = lowpass_map(uc, arr, resol, filter, order=order)
    mask = maskmap_class.mapmask(
        arr=arrlp, uc=uc, kern_rad=kern, prob=prob, itr=itr
    )
    return mask, arrlp


def mask_from_map_connectedpixels(m1, binthresh=None):
    """
    This method generates a mask from a given map based on their
    pixel connectivity. Connectivity is searched on a lowpass map to
    15 A.

    Inputs:
        m1: map object from EMDA2.
        binthresh: binarisation threshold for lowpass map. default is
            max_density_value * 0.1

    Outputs:
        masklist: sorted masks according to their rmsd (largest first)
    """
    from emda2.ext.mapmask import mapmask_connectedpixels

    masklist, lowpassmap = mapmask_connectedpixels(
        m1, binary_threshold=binthresh
    )
    return masklist


def applymask(m1, mm):
    """
    Returns mask applied map as a 3D numpy array

    Inputs:
        m1: map object
        mm: mask object

    Outputs:
        arr: mask applied map as 3D numpy array
    """
    return m1.workarr * mm.workarr


def lowpass_map(
    uc,
    arr1,
    resol,
    filter="ideal",
    order=4,
    bin_idx=None,
    sgrid=None,
    res_arr=None,
):
    """Lowpass filters a map to a specified resolution.

    This function applies a lowpass filter on a map to a specified resolution.

    Arguments:
        Inputs:
            uc: float, 1D array
                Unit cell params.
            arr1: float or complex, 3D numpy array
                Real space map or corresponding Fourier coefficients.
                if Fourier coefficients, the zero-frequency component should be
                at the center of the spectrum.
            resol: float
                Resolution cutoff for lowpass filtering in Angstrom units.
            filter: string, optional
                Fiter type to use in truncating Fourier coefficients.
                Currently, only 'ideal' or 'butterworth' filters can be employed.
                Default type is ideal.
            order: integer, optional
                Order of the Butterworth filter. Default is 4.
            bin_idx: integer, ndarray, optional
                labelled grid of bins by bin number
            sgrid: float, ndarray
                labelled grid of bin by resolution
            res_arr: float, 1D array, optional
                resolution array

        Outputs:
            fmap1: complex, 3D array
                Lowpass filtered Fourier coefficeints.
            map1: float, 3D array
                Lowpass filtered map in image/real space
    """
    import emda2.ext.utils as utils

    if filter == "ideal":
        if bin_idx is None or res_arr is None:
            nbin, res_arr, bin_idx, sgrid = get_binidx(cell=uc, arr=arr1)
        dist = np.sqrt((res_arr - resol) ** 2)
        cbin = np.argmin(dist) + 1
        fmap1, map1 = utils.lowpassmap_ideal(
            fc=arr1, bin_idx=bin_idx, cbin=cbin
        )
    elif filter == "butterworth":
        if sgrid is None or res_arr is None:
            nbin, res_arr, bin_idx, sgrid = get_binidx(cell=uc, arr=arr1)
        dist = np.sqrt((res_arr - resol) ** 2)
        cbin = np.argmin(dist) + 1
        fmap1, map1 = utils.lowpassmap_butterworth(
            fc=arr1, sgrid=sgrid, smax=resol, order=4
        )
    return fmap1, map1


def model2map_gm(
    modelxyz,
    resol,
    dim,
    cell,
    maporigin=None,
    outputpath=None,
    shift_to_boxcenter=False,
):
    """Calculates map from the coordinates

    Arguments:
        Inputs:
            modelxyz: string
                Name of the coordinate file in PDB/CIF
            resol: float
                Resolution to which the map to be calculated.
            dim: list of int
                Dimension/sampling of the map to be calculated.
                e.g. dim=[100, 100, 100]
            cell: list of float
                Unit cell of the map to be calculated in the form
                [a, b, c, alf, bet, gam]
            maporigin: list of int, optional
                Origin of the map to be calculated. default to [0, 0, 0]
            outputpath: string, optional
                Path for auxilliary files.
                default to './emda_gemmifiles'
            shift_to_boxcenter: bool, optional
                If True, the image/molecule is centered in the box.

        Outputs:
            Returns the model-based map as a 3D numpy array.
    """

    from servalcat.utils.model import calc_fc_fft

    if outputpath is None:
        outputpath = os.getcwd()
    outputpath = os.path.join(outputpath, "emda_gemmifiles/")
    print("outputpath: ", outputpath)
    # make director for files for refmac run
    if os.path.exists(outputpath):
        shutil.rmtree(outputpath)
    os.mkdir(outputpath)
    # check for valid sampling:
    for i in range(3):
        if dim[i] % 2 != 0:
            dim[i] += 1
    # check for minimum sampling
    min_pix_size = resol / 2  # in Angstrom
    min_dim = np.asarray(cell[:3], dtype="float") / min_pix_size
    min_dim = np.ceil(min_dim).astype(int)
    for i in range(3):
        if min_dim[i] % 2 != 0:
            min_dim += 1
        if min_dim[0] > dim[0]:
            print("Requested dims: ", dim)
            print("Minimum dims needed (for requested resolution): ", min_dim)
            print(
                "!!! Please lower the requested resolution or increase the"
                " grid dimensions !!!"
            )
            raise SystemExit()
    if shift_to_boxcenter:
        from emda.core.modeltools import shift_to_origin, shift_model

        doc = shift_to_origin(modelxyz)
        doc.write_file(outputpath + "model1.cif")
        modelxyz = outputpath + "model1.cif"
    a, b, c = cell[:3]
    st = gemmi.read_structure(modelxyz)
    st.spacegroup_hm = "P 1"
    st.cell.set(a, b, c, 90.0, 90.0, 90.0)
    st.make_mmcif_document().write_file(outputpath + "model.cif")
    asu_data = calc_fc_fft(
        st=st, d_min=resol, source="electron", mott_bethe=True
    )
    griddata = asu_data.get_f_phi_on_grid(dim)
    griddata_np = (np.array(griddata, copy=False)).transpose()
    modelmap = (np.fft.ifftn(np.conjugate(griddata_np))).real
    if np.sum(np.asarray(modelmap.shape, "int") - np.asarray(dim, "int")) != 0:
        cpix = [cell[i] / shape for i, shape in enumerate(modelmap.shape)]
        tpix = [cell[i] / shape for i, shape in enumerate(dim)]
        modelmap = iotools.resample_data(
            curnt_pix=cpix, targt_pix=tpix, arr=modelmap, targt_dim=dim
        )
    if shift_to_boxcenter:
        maporigin = None  # no origin shift allowed
        modelmap = np.fft.fftshift(modelmap)  # bring modelmap to boxcenter
        # shift model to boxcenter
        doc = shift_model(
            mmcif_file=outputpath + "model.cif", shift=[a / 2, b / 2, c / 2]
        )
        doc.write_file(outputpath + "emda_shifted_model.cif")
    if maporigin is None:
        maporigin = [0, 0, 0]
    else:
        shift_x = maporigin[0]
        shift_y = maporigin[1]
        shift_z = maporigin[2]
        modelmap = np.roll(
            np.roll(np.roll(modelmap, -shift_x, axis=0), -shift_y, axis=1),
            -shift_z,
            axis=2,
        )
    return np.transpose(modelmap)  # it seems the transpose is necessary


def realsp_correlation(
    arr_hf1,
    arr_hf2,
    uc,
    norm=False,
    model=None,
    mask=None,
    kernel=5,
    origin=None,
    axorder=None,
):
    """Calculates local correlation in real/image space.

    Arguments:
        Inputs:
            arr_hf1, arr_hf2: Mandatory
                3D numpy arrays corresponding to halfmap1 and halfmap2
            kernel: optional
                This is a 3D numpy array of n x n x n shape, or the
                radius of integration kernal in pixels. Default is 5.
            norm: bool, optional
                If True, correlation will be carried out on normalized maps.
                Default is False.
            model: Optional
                3D numpy array corresponding to modelbased map.
            mask: Optional
                3D numpy array corresponding to mask
            uc: mandatory
                Unit cell as a 1D numpy array

        Outputs:
            Following maps are written out:
            rcc_halfmap_smax?.mrc - reals space half map local correlation.
            rcc_fullmap_smax?.mrc - correlation map corrected to full map
                using the formula 2 x FSC(half) / (1 + FSC(half)).
            If a model included, then
            rcc_mapmodel_smax?.mrc - local correlation map between model and
                full map.
            rcc_truemapmodel_smax?.mrc - truemap-model correaltion map for
                validation purpose.
            rcc object is output
    """
    from emda2.ext import realsp_local

    rcc = realsp_local.RealspaceLocalCC()
    rcc.arr1 = arr_hf1
    rcc.arr2 = arr_hf2
    rcc.kernel = kernel
    rcc.model = model
    rcc.mask = mask
    rcc.norm = norm
    rcc.uc = uc
    rcc.origin = origin
    rcc.axorder = axorder
    rcc.rcc()
    return rcc


def resample_data(curnt_pix, targt_pix, arr, targt_dim=None):
    """Resamples a 3D data array.

    Arguments:
        Inputs:
            curnt_pix: float list, Current pixel sizes along a, b, c.
            targt_pix: float list, Target pixel sizes along a, b c.
            arr: float, 3D array of map values.
            targt_dim: int list, Target sampling along x, y, z.

        Outputs:
            new_arr: float, 3D array
                Resampled 3D array.
    """
    new_arr = iotools.resample2staticmap(
        curnt_pix=curnt_pix, targt_pix=targt_pix, targt_dim=targt_dim, arr=arr
    )
    return new_arr


def apply_bfactor_to_map(f, bf_arr, uc):
    """Applies an array of B-factors on the map.

    Arguments:
        Inputs:
            f: complex Fourier coefficients (FCs) of map
                FCs are arranges so that the 0th frequency component is
                at the center of the spectrum.
            bf_arr: float, 1D array
                An array/list of B-factors.
            uc: float 1D array
                Unit cell.

        Outputs:
            all_mapout: complex, ndarray
                4D array containing FCs of all maps.
                e.g. all_mapout[:,:,:,i], where i represents map number
                corresponding to the B-factor in bf_arr.
    """
    all_mapout = maptools.apply_bfactor_to_map(fmap=f, bf_arr=bf_arr, uc=uc)
    return all_mapout


def map2mtz(arr, uc, mtzname="map2mtz.mtz", resol=None):
    """Convert map into MTZ format.

    Arguments:
        Inputs:
            arr: float 3D numpy array
                Map values
            uc: float 1D array
                Unit cell
            mtzname: string
                Output MTZ file name. Default is map2mtz.mtz
            resol: float
                Resolution cutoff

        Outputs:
            Outputs MTZ file.
    """
    maptools.map2mtz(arr=arr, uc=uc, mtzname=mtzname, resol=resol)


def mtz2map(mtzname, map_size):
    """Converts an MTZ file into MRC format.

    This function converts data in an MTZ file into a 3D Numpy array.
    It combines amplitudes and phases to form Fourier coefficients.

    Arguments:
        Inputs:
            mtzname: string
                MTZ file name.
            map_size: list
                Shape of output 3D Numpy array as a list of three integers.

        Outputs:
            outarr: float
            3D Numpy array of map values.
    """
    arr, unit_cell = maptools.mtz2map(mtzname=mtzname, map_size=map_size)
    return np.transpose(arr), unit_cell


def half2full(hf1arr, hf2arr):
    """Combines half maps to generate full map.

    Arguments:
        Inputs:
            hf1arr: float 3D numpy array of halfmap 1
            hf2arr: float 3D numpy array of halfmap 2

        Outputs:
            fullmap: float 3D numpy array of fullmap
    """
    f1 = np.fft.fftn(hf1arr)
    f2 = np.fft.fftn(hf2arr)
    return np.real(np.fft.ifftn((f1 + f2) / 2.0))


def mask_from_atomic_model(mapname, modelname, atmrad=3):
    """Generates a mask from atomic coordinates.

    Generates a mask from coordinates. First, atomic positions are
    mapped onto a 3D grid. Second, each atomic position is convluted
    with a sphere whose radius is defined by the atmrad paramter.
    Next, one pixel layer dialtion followed by the smoothening of
    edges.

    Arguments:
        Inputs:
            mapname: string
                Name of the map file. This is needed to get the
                sampling, unit cell and origin for the new mask.
                Allowed formats are - MRC/MAP
            modelname: string
                Atomic model name. Allowed formats are - PDB/CIF
            atmrad: float
                Radius of the sphere to be placed on atomic positions in Angstroms.
                Default is 5 A.

        Outputs:
            mask: float, 3D array
                3D Numpy array of the mask.
            Outputs emda_model_mask.mrc.
    """
    from emda2.ext.maskmap_class import mask_from_coordinates

    mapobj = mask_from_coordinates(
        mapname=mapname,
        modelname=modelname,
        atmrad=atmrad,
    )
    return mapobj


def realsp_correlation_mapmodel(
    uc,
    map,
    model,
    resol,
    kernel=5,
    mask=None,
):
    """Calculates real space local correlation between map and model.

    Arguments:
        Inputs:
            uc: float, Unit cell as 1D array
            map: float, 3D numpy array of map values
            model: float, 3D numpy array of model-based map values
            resol: float, resolution of the model based map.
            kernel: optional
                3D numpy array of n x n x n shape, or the
                radius of integration kernal in pixels. Default is 5.
            mask: optional
                float, 3D numpy array of mask values.

        Outputs:
            mapmodel_rcc: float, 3D numpy array of correlation map.
            kern_rad: integer, radius of the spherical kernel used in
                local correlation calculation.
    """
    from emda2.ext import realsp_local

    _, arr1 = lowpass_map(uc=uc, arr1=map, resol=resol, filter="butterworth")
    mapmodel_rcc, kern_rad = realsp_local.mapmodel_rcc(
        arr1=arr1,
        model_arr=model,
        kernel=kernel,
        mask=mask,
    )
    return mapmodel_rcc, kern_rad


def overlay(
    arrlist,
    pixlist,
    cell,
    origin,
    nocom=False,
    optmethod=None,
    tlist=None,
    qlist=None,
    fitres=None,
    r_only=False,
    t_only=False,
):
    from emda2.ext import utils

    maplist = []
    mask = utils.sphere_mask(arrlist[0].shape[0])
    maplist.append(arrlist[0] * mask)
    for i, arr in enumerate(arrlist[1:], start=1):
        resampled_arr = resample_data(
            curnt_pix=pixlist[i],
            targt_pix=pixlist[0],
            arr=arr,
            targt_dim=arrlist[0].shape,
        )
        maplist.append(resampled_arr * mask)
    from emda2.ext import overlay

    if r_only:
        emmap1 = overlay.EmmapOverlay(map_list=maplist, nocom=True)
    else:
        emmap1 = overlay.EmmapOverlay(map_list=maplist, nocom=nocom)
    emmap1.map_origin = origin
    emmap1.pixsize = pixlist[0]
    emmap1.map_dim = arrlist[0].shape
    emmap1.map_unit_cell = cell
    emmap1.load_maps()
    emmap1.calc_fsc_from_maps()
    if tlist is None:
        tlist = [[0.0, 0.0, 0.0] for _ in range(len(arrlist))]
    if qlist is None:
        qlist = [[1.0, 0.0, 0.0, 0.0] for _ in range(len(arrlist))]
    emmap1, rotmat_list, trans_list = overlay.overlay(
        emmap1=emmap1,
        tlist=tlist,
        qlist=qlist,
        ncycles=100,
        fitres=fitres,
        optmethod=optmethod,
        r_only=r_only,
        t_only=t_only,
    )
    return emmap1, rotmat_list, trans_list


def refine_magnification():
    pass


def refine_axis(
    m1,
    axis,
    symorder,
    fitres=6,
    fitfsc=0.5,
    optmethod="nelder-mead",
    fobj=None,
    t_init=None,
    res_arr=None,
    bin_idx=None,
    nbin=None,
):
    """
    Inputs:
        m1: Map object made with iotools.Map()
        axis: rotation axis of the map (output from Proshade)
        symorder: symmetry order of the rotation axis
        fitres: highest resolution for refinement. default to 6 A.
        fitfsc: threshold FSC for deciding resolution level for the refinement
                default to 0.5
        fobj: object for logfile output
        t_init: initial translation vector to apply. default to x=0, y=0, z0
        optmethod: axis optimisation method, default to Nelder-Mead

    Outputs:
        initial_ax: same as input axis, but normalised
        final_ax: refined rotation axis
        final_t = refined translation (in fractional units as output by EMDA)
    """
    from emda2.ext import axis_refinement

    if fobj is None:
        fobj = open("emda_axis-refine.txt", "w")

    emmap1 = axis_refinement.EmmapOverlay(arr=m1.workarr)
    emmap1.map_unit_cell = m1.workcell
    emmap1.bin_idx = bin_idx
    emmap1.res_arr = res_arr
    emmap1.nbin = nbin
    emmap1.map_dim = m1.workarr.shape
    emmap1.map_unit_cell = m1.workcell
    emmap1.pix = [m1.workcell[i] / sh for i, sh in enumerate(m1.workarr.shape)]
    emmap1.prep_data()
    if t_init is None:
        t_init = [0.0, 0.0, 0.0]
    final_ax, final_t, ax_pos = axis_refinement.axis_refine(
        emmap1=emmap1,
        rotaxis=axis,
        symorder=symorder,
        fitres=fitres,
        fitfsc=fitfsc,
        ncycles=10,
        fobj=fobj,
        t_init=t_init,
        optmethod=optmethod,
    )
    return final_ax, final_t, ax_pos


def get_rotation_center(m1, mm, axis, order, resol):
    from emda2.ext.sym import get_rotation_center

    results = get_rotation_center.get_rotation_center(
        m1=m1, mm=mm, axis=axis, order=order, claimed_res=resol
    )
    rotation_center = results[2]
    return rotation_center


def rebox_by_mask(arr, mask, mask_origin, padwidth=10):
    """
    Rebox a map using provided mask

    Inputs:
        arr: ndarray, map to rebox
        mask: ndarra, mask for reboxing

    Outputs:
        reboxed_map: ndarray, reboxed map
        reboxed_mask: ndarray, reboxed mask
    """
    from emda2.ext.utils import rebox_using_mask

    reboxed_map, reboxed_mask = rebox_using_mask(
        arr=arr, mask=mask, mask_origin=mask_origin, padwidth=padwidth
    )
    return reboxed_map, reboxed_mask


def flip_arr(arr, axis="z"):
    # axis to flip
    try:
        if axis == "x":
            ax = 0
        if axis == "y":
            ax = 1
        if axis == "z":
            ax = 2
        return np.flip(m=arr, axis=ax)
    except Exception as e:
        raise e


def get_pointgroup(
    half1,
    half2,
    resol,
    resol4axref=5.0,
    output_maps=False,
    symaverage=False,
    mask=None,
    axlist=None,
    orderlist=None,
    fsclist=None,
    user_pg=None,
    label=None,
    pmap=None,
):
    """
    Determines the point group of the map

    Inputs:
        half1, half2: required strings
                half maps are required
        resol: required float
                nominal resolution of the map
        mask: optional string
                mask name
        axlist: optional
                Initial axes of [tentative] point group.
                Should be given as X1 Y1 Z1 X2 Y2 Z2 format
        orderlist: optional
                Initial orders of axes.
                Should be given as Order-of-axis1 Order-of-axis2 format
        fsclist: optional
                FSCs of axes.
                Should be given as FSC-of-axis1 FSC-of-axis2 format
        resol4axref: optional float
                resolution for axis refinement default to 5 A.
        output_maps: optional bool
                if True maps will be written out. Dafault to False
        symaverage: optional bool
                if True maps will be symmetry averaged. Default to False
        user_pg: optional string
                user claimed point group for logging
        label: optional string
                label to use for logging and map output

    Outputs:
        Dictionary with all the meta data
    """
    from emda2.ext.sym.symanalysis_pipeline import switch

    if axlist is not None:
        assert len(axlist) // 3 == len(orderlist)
        grouped_axlist = [axlist[i : i + 3] for i in range(0, len(axlist), 3)]
    else:
        grouped_axlist = axlist

    params = {
        "half1": half1,
        "half2": half2,
        "pmap": pmap,
        "resol": resol,
        "mask": mask,
        "resol4refinement": resol4axref,
        "output_maps": output_maps,
        "symaverage": symaverage,
        "axlist": grouped_axlist,
        "orderlist": orderlist,
        "fsclist": fsclist,
        "lowres_cutoff": 10.0,
        "pg_decide_fsc": 0.9,
        "user_pg": user_pg,
        "label": label,
        "fitfsc": 0.1,
        "ncycles": 10,
    }
    results = switch(params)
    return results


def apply_transformation(m1, rotmat=None, trans=None, ibin=None, newdim=None):
    """
    This method applys a translaformation (rotation, translation)
    on the map.

    Inputs:
        m1: mapobject from EMDA/iotools.Map
        rotmat: rotation matrix to apply
        trans: translation to apply (Angstroms)
        ibin:

    Outputs:
        transformed Fourier coefficients
    """
    if rotmat is not None or trans is not None:
        if newdim is None:
            newdim = m1.workarr.shape[0]
        if newdim != m1.workarr.shape[0]:
            current_pixsize = m1.workcell[0] / m1.workarr.shape[0]
            newarr = iotools.resample2staticmap(
                curnt_pix=[current_pixsize for _ in range(3)],
                targt_pix=[current_pixsize for _ in range(3)],
                targt_dim=[newdim, newdim, newdim],
                arr=m1.workarr,
            )
            cell = [current_pixsize * newdim for _ in range(3)]
            for _ in range(3):
                cell.append(0.0)
            f1 = fftshift(fftn(fftshift(newarr)))
        else:
            cell = m1.workcell
            f1 = fftshift(fftn(fftshift(m1.workarr)))

        if rotmat is not None:
            print("rotamt:")
            print(rotmat)
            # check for identity rotation
            # if((rotmat.shape[0] == rotmat.shape[1]) and
            #    np.allclose(rotmat, np.eye(rotmat.shape[0]))):
            nbin, res_arr, bin_idx, sgrid = get_binidx(cell=cell, arr=f1)
            f1 = np.expand_dims(f1, axis=3)
            nx, ny, nz, ncopies = f1.shape
            if ibin is None:
                ibin = nbin
            frs = fcodes2.trilinear_sphere(
                rotmat, f1, bin_idx, 0, ibin, ncopies, nx, ny, nz
            )[:, :, :, 0]
            if trans is not None:
                print("translation: ", trans)
                assert len(trans) == 3
                t = [trans[i] / m1.workcell[i] for i in range(3)]
                st = fcodes2.get_st(nx, ny, nz, t)[0]
                return st * frs, cell
            else:
                return frs, cell
        if rotmat is None and trans is not None:
            assert len(trans) == 3
            t = [trans[i] / m1.workcell[i] for i in range(3)]
            st = fcodes2.get_st(nx, ny, nz, t)[0]
            return st * f1, m1.workcell
        # test
        """ from emda2.ext.bfgs import get_rgi, get_f
        ereal_rgi, eimag_rgi = get_rgi(f1)
        frs = get_f(f1, ereal_rgi, eimag_rgi, rotmat)
        return frs """
    else:
        raise SystemExit("No transformation is given!")


def rotate_map_realspace(m1, rotmat=None, threshold=None):
    print("rotamt:")
    print(rotmat)
    if threshold is not None:
        arr = m1.workarr * (m1.workarr > threshold)
    else:
        arr = m1.workarr
    nx, ny, nz = m1.workarr.shape
    transformed_map = fcodes2.trilinear_map(rotmat, arr, 0, nx, ny, nz)
    return transformed_map


def mask_from_halfmaps(h1, h2, emdbid="rhovar"):
    import emda2.ext.mapmask_using_halfmaps as hfmask

    mask = hfmask.main(h1=h1, h2=h2, emdbid=emdbid)
    return [mask]


def emda_weightedmap(h1, h2, bin_idx, nbin, res_arr, B=None):
    import emda2.ext.emda_weightedmap as nem

    weightedmap = nem.bestmap(
        f1=np.fft.fftshift(np.fft.fftn(h1.workarr)), 
        f2=np.fft.fftshift(np.fft.fftn(h2.workarr)), 
        mode=1, 
        bin_idx=bin_idx, 
        nbin=nbin, 
        res_arr=res_arr, 
        B=B)
    return np.real(np.fft.ifftn(np.fft.ifftshift(weightedmap)))


""" def symmetry_average_map(mlist, axlist, folds, rot_centre=None):
    from emda2.ext.sym.average_symcopies import average
    if rot_centre is None:
        t = [0., 0., 0.]
    else:

    m1 = mlist[0]
    nbin, res_arr, bin_idx, sgrid = get_binidx(cell=m1.workcell, arr=m1.workarr)
    f_avg_list = []
    for i, mp in enumerate(mlist):
        fo = fftshift(fftn(fftshift(mp.workarr)))
        f_avg = average(fo=fo, 
            axis=axlist[i], fold=folds[i], t=tlist[i],bin_idx=bin_idx, ibin=nbin)
        f_avg_list.append(f_avg)
    return f_avg_list """


if __name__ == "__main__":
    """imap = "/Users/ranganaw/MRC/REFMAC/ligand_challenge/EMD-7770/Trimmed_maps_and_model/emd_7770_trimmed.mrc"
    mapobj = iotools.Map(name=imap)
    mapobj.read()
    nbin, res_arr, bin_idx = get_binidx(mapobj.cell, mapobj.arr)
    power_spectrum = get_map_power(fo=np.fft.fftshift(np.fft.fftn(mapobj.arr)),
                bin_idx=bin_idx, nbin=nbin)
    print("Resolution   bin     Power")
    for i in range(len(res_arr)):
        print("{:.2f} {:.4f}".format(res_arr[i], power_spectrum[i]))
    exit()"""

    """ imap1 = "/Users/ranganaw/MRC/REFMAC/ligand_challenge/EMD-7770/Trimmed_maps_and_model/emd_7770_half_map_1_trimmed.mrc"
    imap2 = "/Users/ranganaw/MRC/REFMAC/ligand_challenge/EMD-7770/Trimmed_maps_and_model/emd_7770_half_map_2_trimmed.mrc"
    #imap1 = "/Users/ranganaw/MRC/REFMAC/ligand_challenge/EMD-7770/EMD-7770/emd_7770_half1.map"
    #imap2 = "/Users/ranganaw/MRC/REFMAC/ligand_challenge/EMD-7770/EMD-7770/emd_7770_half2.map"
    mapobj1 = iotools.Map(name=imap1)
    mapobj1.read()
    mapobj2 = iotools.Map(name=imap2)
    mapobj2.read() """

    """ nbin, res_arr, bin_idx = get_binidx(mapobj1.cell, mapobj1.arr)
    twomapfsc = fsc(f1=np.fft.fftshift(np.fft.fftn(mapobj1.arr)),
                    f2=np.fft.fftshift(np.fft.fftn(mapobj2.arr)),
                    bin_idx=bin_idx,
                    nbin=nbin)

    plotter.plot_nlines(res_arr=res_arr,
                        list_arr=[twomapfsc],
                        mapname="twomap_fsc.eps",) """

    """ mask = mask_from_halfmaps(uc=mapobj1.cell,
                              half1=mapobj1.arr,
                              half2=mapobj2.arr,)
    mobj = iotools.Map()
    mobj.arr = mask
    mobj.cell = mapobj1.cell
    mobj.origin = mapobj1.origin
    mobj.axorder = mapobj1.axorder
    mobj.write(filename='mask.map') """

    # finding pointgroup
    half1 = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21231/emd_21231_half_map_1.map"
    mask = "/Users/ranganaw/MRC/REFMAC/symmetry/EMD-21231/emd_21231_msk_1.map"
    resol = 6
    get_pointgroup(half1, mask, resol)
