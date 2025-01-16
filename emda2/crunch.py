import numpy as np
import math

def get_resol(uc, h, k, l):
    a = uc[0]
    b = uc[1]
    c = uc[2]
    vol = a * b * c
    sa = b * c / vol
    sb = a * c / vol
    sc = a * b / vol
    s2 = ((h * sa) ** 2 + (k * sb) ** 2 + (l * sc) ** 2) / 4.0
    if s2 == 0.0:
        s2 = 1.0e-10
    tmp = math.sqrt(s2)
    return 1.0 / (2.0 * tmp)

def resol_grid_em(uc, mode, nx, ny, nz):
    maxbin = np.amax(np.array([nx, ny, nz]) // 2)
    bin_idx = np.full((nz, ny, nx), -100, dtype=int)
    s_grid = np.zeros((nz, ny, nx), dtype=float)
    res_arr = np.zeros(maxbin, dtype=float)
    nbin = 0

    debug = (mode == 1)

    if debug: print('fcodes2...')

    r = np.zeros(3, dtype=float)
    xyzmin = np.zeros(3, dtype=int)
    xyzmax = np.zeros(3, dtype=int)
    hkl = np.zeros(3, dtype=int)

    nxyz = np.array([nx, ny, nz], dtype=int)

    xyzmin[0] = -nxyz[0] // 2
    xyzmin[1] = -nxyz[1] // 2
    xyzmin[2] = -nxyz[2] // 2
    xyzmax = -(xyzmin + 1)

    if debug: 
        print('xyzmin = ', xyzmin)
        print('xyzmax = ', xyzmax[0], xyzmax[1], 0)
        print('unit cell = ', uc)

    r[0] = get_resol(uc, float(xyzmax[0]), 0.0, 0.0)
    r[1] = get_resol(uc, 0.0, float(xyzmax[1]), 0.0)
    r[2] = get_resol(uc, 0.0, 0.0, float(xyzmax[2]))

    if debug: print('a-max, b-max, c-max = ', r)

    sloc = np.argmin(r)
    hkl[sloc] = 1

    step_indices = np.arange(xyzmax[sloc]) + 1.5
    steps = step_indices[:, np.newaxis] * hkl
    resols = np.array([get_resol(uc, *step) for step in steps])

    # Ensure we do not exceed maxbin
    nbin = min(len(resols), maxbin)
    res_arr[:nbin] = resols[:nbin]

    if debug: print('nbin=', nbin)

    high_res = res_arr[nbin - 1]
    low_res = get_resol(uc, 0.0, 0.0, 0.0)

    if debug: print(f"Low res={low_res} High res={high_res} 'A'")

    bin_arr = step_indices[:nbin]

    x = np.arange(xyzmin[0], xyzmax[0] + 1)
    y = np.arange(xyzmin[1], xyzmax[1] + 1)
    z = np.arange(xyzmin[2], xyzmax[2] + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    distances = np.sqrt(xx**2 + yy**2 + zz**2)

    # print(bin_arr)
    # print(distances)

    for ib in range(nbin):
        mask = distances <= bin_arr[ib]
        bin_idx[mask] = ib
        s_grid[mask] = 1.0 / res_arr[ib]

    return bin_idx, s_grid, res_arr, nbin

# Example usage:
uc = np.array([10, 10, 10, 90, 90, 90], dtype=float)
mode = 0
nx, ny, nz = 600, 600, 600

# Benchmark the Python code
import timeit
time_taken = timeit.timeit('resol_grid_em(uc, mode, nx, ny, nz)', globals=globals(), number=1)
print(f"Python code execution time: {time_taken} seconds")

bin_idx, s_grid, res_arr, nbin = resol_grid_em(uc, mode, nx, ny, nz)

# print("bin_idx:", bin_idx)
# print("s_grid:", s_grid)
# print("res_arr:", res_arr)
# print("nbin:", nbin)
