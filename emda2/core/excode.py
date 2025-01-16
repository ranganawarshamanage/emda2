import fcodes2 as fcode
from typing import List, Tuple
import numpy as np

debug_mode = 0

def get_st(dim: Tuple[int, int, int], t: List):
    nx, ny, nz = dim
    st, _, _, _ = fcode.get_st(nx, ny, nz, t)
    return st
    
def make_resgrid(dim: Tuple[int, int, int], uc: List):
    nx, ny, nz = dim
    maxbin = np.amax(np.array([nx // 2, ny // 2, nz // 2]))
    if nx == ny == nz:
        nbin, res_arr, bin_idx, sgrid = fcode.resol_grid_em(
            uc, debug_mode, maxbin, nx, ny, nz
        )
    else:
        nbin, res_arr, bin_idx, sgrid = fcode.resolution_grid(
            uc, debug_mode, maxbin, nx, ny, nz
        )
    return bin_idx, sgrid, res_arr[:nbin]
    
def make_resgrid_from_resarr(dim, uc, res_arr):
    nx, ny, nz = dim
    bin_idx, sgrid = fcode.resolution_grid_from_given_resarr(
        uc,
        res_arr,
        debug_mode,
        len(res_arr),
        nx,
        ny,
        nz
    )
    return bin_idx, sgrid

def make_resarr(uc, maxbin, firststep=2.5):       
    res_arr = fcode.make_resarr(
         uc,
         maxbin,
         firststep,
    )
    return res_arr

def conv3d_to_1d
        subroutine conv3d_to_1d(f3d,uc,nx,ny,nz,mode,f1d,resol1d) ! in :fcodes:fcodes.f90
            complex*16 dimension(1.0 * nx,1.0 * ny,1.0 * nz),intent(in) :: f3d
            real dimension(6),intent(in) :: uc
            integer, optional,intent(in),check((shape(f3d,0))/(1.0)==nx),depend(f3d) :: nx=(shape(f3d,0))/(1.0)
            integer, optional,intent(in),check((shape(f3d,1))/(1.0)==ny),depend(f3d) :: ny=(shape(f3d,1))/(1.0)
            integer, optional,intent(in),check((shape(f3d,2))/(1.0)==nz),depend(f3d) :: nz=(shape(f3d,2))/(1.0)
            integer intent(in) :: mode
            complex*16 dimension(nx*ny*(nz+2)/2),intent(out),depend(nx,ny,nz) :: f1d
            real dimension(nx*ny*(nz+2)/2),intent(out),depend(nx,ny,nz) :: resol1d
        end subroutine conv3d_to_1d
    
def 