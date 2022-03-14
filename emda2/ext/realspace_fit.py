# realspace_fit
import numpy as np
from emda2.core import iotools
import emda2.emda_methods2 as em
from emda import core
import emda.emda_methods as em1
import emda2.core as core2


def get_arr(arr, myrgi, rotmat):
    nx, ny, nz = arr.shape
    points = core2.maptools.get_points(rotmat,nx,ny,nz)
    return myrgi(points).reshape(nx,ny,nz)


class Bfgs:
    def __init__(self):
        self.method = "BFGS"
        self.m0 = None
        self.m1 = None
        self.mrt = self.m1
        self.q = np.array([1., 0., 0., 0.], 'float')
        self.t = np.array([0., 0., 0.], 'float')

    #3. functional
    def functional(self, x):
        nx, ny, nz = self.m1.shape
        step = np.asarray(x[3:], 'float')
        self.q = np.array([1., 0., 0., 0.], 'float') + np.insert(step, 0, 0.0)
        self.q = self.q / np.sqrt(np.dot(self.q, self.q))
        rotmat = core.quaternions.get_RM(self.q)
        mrt = get_arr(self.m1, self.myrgi, rotmat)
        self.t = x[:3]
        self.mrt = em1.shift_density(mrt, shift=self.t)
        fval = np.sum(self.m0 * self.mrt) / (nx*ny*nz) # divide by vol is to scale
        print('fval, q, t ', fval.real, self.q, self.t)
        return -fval.real

    #3. optimize using BFGS
    def optimize(self):
        from scipy.optimize import minimize
        self.myrgi = core2.maptools.interp_rgi(data=self.m1)
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'float')
        options = {'maxiter': 400}
        print("Optimization methos: ", self.method)
        result = minimize(fun=self.functional, x0=x, method='Nelder-Mead', tol=1e-5, options=options)
        print(result)
        print("Final q: ", self.q)  
        print("Final t: ", self.t)
        print("Euler angles: ", np.rad2deg(core.quaternions.rotationMatrixToEulerAngles(core.quaternions.get_RM(self.q))))     
        print("Outputting fitted map...")
        rotmat = core.quaternions.get_RM(self.q)





def main():
    maplist = [    
        "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/newtest/lig_full.mrc",
        "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/newtest/ligfull_rotated_2fold_refined.mrc"
        #"/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/newtest/newmap2.mrc"
            ]

    masklist = [
        "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/newtest/halfmap_mask.mrc",
        "/Users/ranganaw/MRC/REFMAC/Vinoth/reboxed_maps/symmetrise/pointgroup/apply_transformation_on_halfmaps/newtest/ligfull_rotated_2fold_refined_mask.mrc"
    ]

    arrlist = []
    pixlist = []
    cells = []
    origins = []
    try:
        for imap, imask in zip(maplist, masklist):
            mp = iotools.Map(imap)
            mp.read()
            mk = iotools.Map(imask)
            mk.read()
            arrlist.append(mp.arr * mk.arr)
            cells.append(mp.cell)
            origins.append(mp.origin)
            pixlist.append([mp.cell[j] / dim for j, dim in enumerate(mp.arr.shape)])        
    except:
        for imap in maplist:
            mp = iotools.Map(name=imap)
            mp.read()
            arrlist.append(mp.arr)
            cells.append(mp.cell)
            origins.append(mp.origin)
            pixlist.append([mp.cell[j] / dim for j, dim in enumerate(mp.arr.shape)])

    from emda2.ext import utils
    maplist = []
    mask = utils.sphere_mask(arrlist[0].shape[0])
    #maplist.append(arrlist[0] * mask)
    opt = Bfgs()
    opt.m0 = arrlist[0] * mask
    opt.m1 = arrlist[1] * mask
    opt.method = 'nelder-mead'
    opt.optimize()

    
if __name__=="__main__":
    main()