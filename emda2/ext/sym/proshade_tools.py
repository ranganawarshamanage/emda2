"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
import traceback
from emda2.core import emdalogger

# Run codes for proshade

try:
    import proshade
except ImportError:
    print(
        "Proshade not found or could bot be imported!\nTry installing from"
        " https://github.com/michaltykac/proshade.git"
    )
    raise SystemExit()


def proshade_overlay(map1, map2, fitresol=4.0):
    ps = proshade.ProSHADE_settings()
    ps.task = proshade.OverlayMap
    ps.verbose = -1
    ps.setResolution(fitresol)
    ps.addStructure(map1)
    ps.addStructure(map2)
    rn = proshade.ProSHADE_run(ps)
    eulerAngles = rn.getEulerAngles()
    rotMatrix = rn.getOptimalRotMat()
    return rotMatrix


def get_symmops_from_proshade(mapname, resolution=8.0, fobj=None):
    try:
        # Create ProSHADE settings object
        ps = proshade.ProSHADE_settings()

        # Set up the run
        ps.task = proshade.Symmetry
        ps.verbose = -1
        ps.addStructure(mapname)
        ps.setResolution(resolution)
        ps.setMapResolutionChange(True)
        ps.setMapCentering(True)
        ps.setSymmetryCentreSearch(False)
        # ps.usePhase = False  # to get the phaseless rotation function.

        # Log Proshade is running
        emdalogger.log_string(fobj, f"Proshade is running on {mapname}")

        # Run ProSHADE
        rn = proshade.ProSHADE_run(ps)

        # Retrieve results
        recSymmetryType = rn.getSymmetryType()
        recSymmetryFold = rn.getSymmetryFold()
        recSymmetryAxes = rn.getAllCSyms()

        proshade_pg = f"{recSymmetryType}{recSymmetryFold}"

        if len(recSymmetryAxes) > 0:
            if fobj is not None:
                emdalogger.log_string(fobj, f"Proshade results for {mapname}")
                emdalogger.log_string(
                    fobj,
                    f"Fold\tx\ty\tz\tAngle\tHeight\tAvg.FSC@{resolution}A"
                )
                for axis in recSymmetryAxes:
                    emdalogger.log_string(
                        fobj,
                        f"{axis[0]:<2}\t{axis[1]:+1.3f}\t{axis[2]:+1.3f}\t"
                        f"{axis[3]:+1.3f}\t{axis[4]:+1.3f}\t"
                        f"{axis[5]:+1.4f}\t{axis[6]:1.3f}"
                    )
                emdalogger.log_string(
                    fobj, f"Proshade pointgroup: {proshade_pg}")

                # Extracting data for return
                fold, x, y, z, theta, peakh, afsc = zip(*recSymmetryAxes)
                return [fold, x, y, z, peakh, proshade_pg, afsc]

        else:
            if fobj is not None:
                fobj.write(f"Proshade gave empty axlist on {mapname}\n")
            print(f"Proshade gave empty axlist on {mapname}\n")
            return []

    except RuntimeError as e:
        if fobj is not None:
            fobj.write(f"Proshade fail on {mapname}\n")
            fobj.write(traceback.format_exc())
        print(f"Proshade fail on {mapname}\n")
        print(e)


def process_proshade_results(proshade_results):
    axis_list, orderlist = [], []
    nrecords = len(proshade_results[0])
    orderlist = proshade_results[0]
    fsclist = proshade_results[-1]
    x = proshade_results[1]
    y = proshade_results[2]
    z = proshade_results[3]
    for i in range(nrecords):
        axis_list.append([x[i], y[i], z[i]])
    return axis_list, orderlist, fsclist
