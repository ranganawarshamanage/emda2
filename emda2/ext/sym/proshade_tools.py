"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
import traceback

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


def get_symmops_from_proshade(mapname, fobj=None):
    """Create the settings object"""
    ps = proshade.ProSHADE_settings()
    """ Set up the run """
    ps.task = proshade.Symmetry
    ps.verbose = -1
    # ps.setResolution(8.0)
    ps.addStructure(mapname)
    # new settings from Michal
    ps.setResolution(8.0)
    ps.setMapResolutionChange(True)
    ps.setMapCentering(True)
    ps.setSymmetryCentreSearch(False)
    # ps.usePhase = False # to get the phaseless rotation function.
    
    try:
        """ Run ProSHADE """
        rn = proshade.ProSHADE_run(ps)
        """ Retrieve results """
        recSymmetryType = rn.getSymmetryType()
        recSymmetryFold = rn.getSymmetryFold()
        recSymmetryAxes = rn.getAllCSyms()

        print(
            "Detected "
            + str(recSymmetryType)
            + "-"
            + str(recSymmetryFold)
            + " symetry."
        )
        proshade_pg = str(recSymmetryType) + str(recSymmetryFold)
        print("Proshade point group: ", proshade_pg)
        if len(recSymmetryAxes) > 0:
            print(
                "Fold      x         y         z       Angle     Height    "
                " Avg. FSC"
            )
            for iter in range(0, len(recSymmetryAxes)):
                print(
                    "  %s    %+1.3f    %+1.3f    %+1.3f    %+1.3f    %+1.4f   "
                    " %1.3f"
                    % (
                        recSymmetryAxes[iter][0],
                        recSymmetryAxes[iter][1],
                        recSymmetryAxes[iter][2],
                        recSymmetryAxes[iter][3],
                        recSymmetryAxes[iter][4],
                        recSymmetryAxes[iter][5],
                        recSymmetryAxes[iter][6],
                    )
                )
            if fobj is not None:
                fobj.write("Proshade results for %s\n" % mapname)
                for iter in range(0, len(recSymmetryAxes)):
                    fobj.write(
                        "  %s    %+1.3f    %+1.3f    %+1.3f    %+1.3f    %+1.4f   "
                        " %1.3f\n"
                        % (
                            recSymmetryAxes[iter][0],
                            recSymmetryAxes[iter][1],
                            recSymmetryAxes[iter][2],
                            recSymmetryAxes[iter][3],
                            recSymmetryAxes[iter][4],
                            recSymmetryAxes[iter][5],
                            recSymmetryAxes[iter][6],
                        )
                    )
                fobj.write("Proshade pointgroup %s\n" % proshade_pg)

            fold, x, y, z, theta, peakh, afsc = [], [], [], [], [], [], []
            for row in recSymmetryAxes:
                fold.append(int(row[0]))
                x.append(row[1])
                y.append(row[2])
                z.append(row[3])
                theta.append(row[4])
                peakh.append(row[5])
                afsc.append(row[6])
            return [fold, x, y, z, peakh, proshade_pg, afsc]
        else:
            if fobj is not None:
                fobj.write("Proshade gave empty axlist on %s\n" % mapname)
            print("Proshade gave empty axlist on %s\n" % mapname)
            return []
    except RuntimeError as e:
        if fobj is not None:
            fobj.write("Proshade fail on %s\n" % mapname)
            fobj.write(traceback.format_exc())
        print("Proshade fail on %s\n" % mapname)
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
