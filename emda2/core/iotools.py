
"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import numpy as np
import mrcfile as mrc

def test():
    """ Tests iotools module installation. """

    print("iotools test ... Passed")

class Map():
    def __init__(self, name):
        self.name = name
        self.arr = None
        self.cell = None
        self.origin = None
        self.axorder = None
        self.paddedarr = None
        self.newcell = None

    def read(self):
        try:
            file = mrc.open(self.name)
            axorder = (file.header.mapc-1, file.header.mapr-1, file.header.maps-1)
            self.axorder = axorder
            #print('order: ', order)
            axes_order = "".join(["XYZ"[i] for i in axorder])
            print('Axes order: ', axes_order)
            self.arr = np.moveaxis(a=np.asarray(file.data, dtype="float"), 
                              source=(2,1,0), 
                              destination=axorder)
            unit_cell = np.zeros(6, dtype='float')
            cell = file.header.cella[['x', 'y', 'z']]
            unit_cell[:3] = cell.astype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')]).view(('<f4',3))
            unit_cell[3:] = float(90)
            self.cell = unit_cell
            self.origin = [
                    1 * file.header.nxstart,
                    1 * file.header.nystart,
                    1 * file.header.nzstart,
                ]
            file.close()
            print(self.name, self.arr.shape, self.cell[:3])
            self.preprocess_read()
        except FileNotFoundError as e:
            print(e)

    def write(self):
        if self.name == "":
            filename="new.mrc"
        else:
            filename=self.name
        if self.axorder is None:
            self.axorder = (0,1,2)
        if self.origin is None:
            self.origin = [0.0, 0.0, 0.0]
        self.arr = np.moveaxis(a=self.arr, 
                              source=self.axorder, 
                              destination=(2,1,0))
        file = mrc.new(name=filename, 
                       data=np.float32(self.arr), 
                       compression=None, 
                       overwrite=True)
        file.header.cella.x = self.cell[0]
        file.header.cella.y = self.cell[1]
        file.header.cella.z = self.cell[2]
        file.header.nxstart = self.origin[0]
        file.header.nystart = self.origin[1]
        file.header.nzstart = self.origin[2]
        file.close()

    def preprocess_read(self):
        arr = self.arr
        pixsize = self.cell[0] / self.arr.shape[0]
        tdim = [nd+1 if nd%2!=0 else nd for nd in self.arr.shape]
        self.workcell = np.asarray([pixsize*dim for dim in tdim] + [90. for _ in range(3)], 'float')
        self.workarr = padimage(arr, tdim)


def write_3d2mtz(unit_cell, mapdata, outfile="map2mtz.mtz", resol=None):
    """ Writes 3D Numpy array into MTZ file.

    Arguments:
        Inputs:
            unit_cell: float, 1D array
                Unit cell params.
            mapdata: complex, 3D array
                Map values to write.
            resol: float, optional
                Map will be output for this resolution.
                Default is up to Nyquist.

        Outputs: 
            outfile: string
            Output file name. Default is map2mtz.mtz.
    """
    from emda2.core.mtz import write_3d2mtz

    write_3d2mtz(unit_cell=unit_cell,
                 mapdata=mapdata,
                 outfile=outfile,
                 resol=resol)

def resample2staticmap(curnt_pix, targt_pix, targt_dim, arr, sf=False, fobj=None):
    """Resamples a 3D array.

    Arguments:
        Inputs:
            curnt_pix: float list, Current pixel sizes along a, b, c.
            targt_pix: float list, Target pixel sizes along a, b c.
            targt_dim: int list, Target sampling along x, y, z.
            arr: float, 3D array of map values.
            sf: bool, optional
                If True, returns a complex array. Otherwise, float array
            fobj: optional. Logger file object

        Outputs:
            new_arr: float, 3D array
                Resampled 3D array. If sf was used, return is a complex array
    """
    # Resamling arr into an array having target_dim
    if targt_dim is not None:
        tnx, tny, tnz = targt_dim
    else:
        targt_dim = arr.shape
        tnx, tny, tnz = targt_dim
    nx, ny, nz = arr.shape
    if len(curnt_pix) < 3:
        curnt_pix.append(curnt_pix[0])
        curnt_pix.append(curnt_pix[0])
    if len(targt_pix) < 3:
        targt_pix.append(targt_pix[0])
        targt_pix.append(targt_pix[0])
    print("Current pixel size: ", curnt_pix)
    print("Target pixel size: ", targt_pix)
    if fobj is not None:
        fobj.write(
            "pixel size [current, target]: "
            + str(curnt_pix)
            + " "
            + str(targt_pix)
            + " \n"
        )
    if np.all(abs(np.array(curnt_pix) - np.array(targt_pix)) < 1e-3):
        dx = (tnx - nx) // 2
        dy = (tny - ny) // 2
        dz = (tnz - nz) // 2
        if dx == dy == dz == 0:
            print("No change of dims")
            if fobj is not None:
                fobj.write("No change of dims \n")
            new_arr = arr
        if np.any(np.array([dx, dy, dz]) > 0):
            print("Padded with zeros")
            if fobj is not None:
                fobj.write("Padded with zeros \n")
            new_arr = padimage(arr, targt_dim)
        if np.any(np.array([dx, dy, dz]) < 0):
            print("Cropped image")
            if fobj is not None:
                fobj.write("Cropped image \n")
            new_arr = cropimage(arr, targt_dim)
    else:
        newsize = []
        print("arr.shape: ", arr.shape)
        for i in range(3):
            ns = int(round(arr.shape[i] * (curnt_pix[i] / targt_pix[i])))
            newsize.append(ns)
        print("Resizing in Fourier space and transforming back")
        if fobj is not None:
            fobj.write("Resizing in Fourier space and transforming back \n")
        new_arr = resample(arr, newsize, sf=False)
        if np.any(np.array(new_arr.shape) < np.array(targt_dim)):
            print("pading image...")
            new_arr = padimage(new_arr, targt_dim)
        elif np.any(np.array(new_arr.shape) > np.array(targt_dim)):
            print("cropping image...")
            new_arr = cropimage(new_arr, targt_dim)
    return new_arr

def resample_on_anothermap(uc1, uc2, arr1, arr2):
    # arr1 is taken as reference and arr2 is resampled on arr1
    tpix1 = uc1[0] / arr1.shape[0]
    tpix2 = uc1[1] / arr1.shape[1]
    tpix3 = uc1[2] / arr1.shape[2]
    dim1 = int(round(uc2[0] / tpix1))
    dim2 = int(round(uc2[1] / tpix2))
    dim3 = int(round(uc2[2] / tpix3))
    target_dim = arr1.shape
    arr2 = resample(arr2, [dim1, dim2, dim3], sf=False)
    if np.any(np.array(arr2.shape) < np.array(target_dim)):
        arr2 = padimage(arr2, target_dim)
    elif np.any(np.array(arr2.shape) > np.array(target_dim)):
        arr2 = cropimage(arr2, target_dim)
    return arr2


def padimage(arr, tdim):
    if len(tdim) == 3:
        tnx, tny, tnz = tdim
    elif len(tdim) < 3:
        tnx = tny = tnz = tdim[0]
    else:
        raise SystemExit("More than 3 dimensions given. Cannot handle")
    #print("current shape: ", arr.shape)
    #print("target shape: ", tdim)
    nx, ny, nz = arr.shape
    assert tnx >= nx
    assert tny >= ny
    assert tnz >= nz
    dz = (tnz - nz) // 2 + (tnz - nz) % 2
    dy = (tny - ny) // 2 + (tny - ny) % 2
    dx = (tnx - nx) // 2 + (tnx - nx) % 2
    #print(dx, dy, dz)
    image = np.zeros((tnx, tny, tnz), arr.dtype)     
    xstart, ystart, zstart = [0 if px==1 else px for px in [dx, dy, dz]] 
    xend, yend, zend = xstart + nx, ystart + ny, zstart + nz
    image[xstart:xend, ystart:yend, zstart:zend] = arr
    #image[-(nx + dx):-dx, -(ny + dy):-dy, -(nz + dz):-dz] = arr
    #image[dx:nx+dx, dy:ny+dy, dz:nz+dz] = arr
    return image

def cropimage(arr, tdim):
    if len(tdim) == 3:
        tnx, tny, tnz = tdim
    elif len(tdim) < 3:
        tnz = tny = tnx = tdim[0]
    else:
        raise SystemExit("More than 3 dimensions given. Cannot handle")
    nx, ny, nz = arr.shape
    #print("Current dim [nx, ny, nz]: ", nx, ny, nz)
    #print("Target dim [nx, ny, nz]: ", tnx, tny, tnz)
    assert tnx <= nx
    assert tny <= ny
    assert tnz <= nz
    dx = abs(nx - tnx) // 2
    dy = abs(ny - tny) // 2
    dz = abs(nz - tnz) // 2
    return arr[dx: tdim[0] + dx, dy: tdim[1] + dy, dz: tdim[2] + dz]


def resample(x, newshape, sf):
    xshape = list(x.shape)
    for i in range(3):
        if x.shape[i] % 2 != 0:
            xshape[i] += 1
        if newshape[i] % 2 != 0:
            newshape[i] += 1
    temp = np.zeros(xshape, x.dtype)
    temp[:x.shape[0], :x.shape[1], :x.shape[2]] = x
    x = temp
    print(np.array(x.shape) - np.array(newshape))
    # nosampling
    if np.all((np.array(x.shape) - np.array(newshape)) == 0):
        print('no sampling')
        if sf:
            return np.fft.fftshift(np.fft.fftn(x))
        else:
            return x
    # Forward transform
    X = np.fft.fftn(x)
    X = np.fft.fftshift(X)
    # Placeholder array for output spectrum
    Y = np.zeros(newshape, X.dtype)
    # upsampling
    dx = []
    if np.any((np.array(x.shape) - np.array(newshape)) < 0):
        print('upsampling...')
        for i in range(3):
            dx.append(abs(newshape[i] - X.shape[i]) // 2)
        Y[dx[0]: dx[0] + X.shape[0], 
          dx[1]: dx[1] + X.shape[1], 
          dx[2]: dx[2] + X.shape[2]] = X
    # downsampling
    if np.any((np.array(x.shape) - np.array(newshape)) > 0):
        print('downsampling...')
        for i in range(3):
            dx.append(abs(newshape[i] - X.shape[i]) // 2)
        Y[:, :, :] = X[
                    dx[0]: dx[0] + newshape[0], 
                    dx[1]: dx[1] + newshape[1], 
                    dx[2]: dx[2] + newshape[2]
                    ]
    if sf:
        return Y
    Y = np.fft.ifftshift(Y)
    return (np.fft.ifftn(Y)).real



def read_mmcif(mmcif_file):
    """Reading mmcif using gemmi and output as numpy 1D arrays"""
    import gemmi
    # 
    doc = gemmi.cif.read_file(mmcif_file)
    block = doc.sole_block()  # cif file as a single block
    a = block.find_value("_cell.length_a")
    b = block.find_value("_cell.length_b")
    c = block.find_value("_cell.length_c")
    alf = block.find_value("_cell.angle_alpha")
    bet = block.find_value("_cell.angle_beta")
    gam = block.find_value("_cell.angle_gamma")
    cell = np.array([a, b, c, alf, bet, gam], dtype="float")
    # Reading X coordinates in all atoms
    col_x = block.find_values("_atom_site.Cartn_x")
    col_y = block.find_values("_atom_site.Cartn_y")
    col_z = block.find_values("_atom_site.Cartn_z")
    # Reading B_iso values
    col_Biso = block.find_values("_atom_site.B_iso_or_equiv")
    # Casting gemmi.Columns into a numpy array
    x_np = np.array(col_x, dtype="float", copy=False)
    y_np = np.array(col_y, dtype="float", copy=False)
    z_np = np.array(col_z, dtype="float", copy=False)
    Biso_np = np.array(col_Biso, dtype="float", copy=False)
    return cell, x_np, y_np, z_np, Biso_np


def run_refmac_sfcalc(filename, resol, lig=True, bfac=None, ligfile=None):
    import os
    import os.path
    import subprocess
    #
    current_path = os.getcwd() # get current path
    filepath = os.path.abspath(os.path.dirname(filename)) + '/'
    os.chdir(filepath)
    fmtz = filename[:-4] + ".mtz"
    cmd = ["refmac5", "XYZIN", filename, "HKLOUT", fmtz]
    if ligfile is not None:
        cmd = ["refmac5", "XYZIN", filename, "HKLOUT", fmtz, "lib_in", ligfile]
        lig = False
    # Creating the sfcalc.inp with custom parameters (resol, Bfac)
    sfcalc_inp = open(filepath+"sfcalc.inp", "w+")
    sfcalc_inp.write("mode sfcalc\n")
    sfcalc_inp.write("sfcalc cr2f\n")
    if lig:
        sfcalc_inp.write("make newligand continue\n")
    sfcalc_inp.write("resolution %f\n" % resol)
    if bfac is not None and bfac > 0.0:
        sfcalc_inp.write("temp set %f\n" % bfac)
    sfcalc_inp.write("source em mb\n")
    sfcalc_inp.write("make hydrogen yes\n")
    sfcalc_inp.write("end")
    sfcalc_inp.close()
    # Read in sfcalc_inp
    PATH = filepath+"sfcalc.inp"
    logf = open(filepath+"sfcalc.log", "w+")
    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print("sfcalc.inp exists and is readable")
        inp = open(filepath+"sfcalc.inp", "r")
        # Run the command with parameters from file f2mtz.inp
        subprocess.call(cmd, stdin=inp, stdout=logf)
        logf.close()
        inp.close()
    else:
        raise SystemExit("File is either missing or not readable")
    os.chdir(current_path)


def read_atomsf(atm, fpath=None):
    # outputs A and B coefficients corresponding to atom(atm)
    found = False
    with open(fpath) as myFile:
        for num, line in enumerate(myFile, 1):
            if line.startswith(atm):
                found = True
                break
    A = np.zeros(5, dtype=np.float)
    B = np.zeros(5, dtype=np.float)
    if found:
        ier = 0
        f = open(fpath)
        all_lines = f.readlines()
        for i in range(4):
            if i == 0:
                Z = all_lines[num + i].split()[0]
                NE = all_lines[num + i].split()[1]
                A[i] = all_lines[num + i].split()[-1]
                B[i] = 0.0
            elif i == 1:
                A[1:] = np.asarray(all_lines[num + i].split(), dtype=np.float)
            elif i == 2:
                B[1:] = np.asarray(all_lines[num + i].split(), dtype=np.float)
        f.close()
    else:
        ier = 1
        Z = 0
        NE = 0.0
        print(atm, "is not found!")
    return int(Z), float(NE), A, B, ier


if __name__=="__main__":
    imap = sys.argv[1:][0]

    print(imap)



