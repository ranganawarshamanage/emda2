"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

import gzip, shutil
import xml.etree.ElementTree as ET
import os

def fetch_halfmaps(emdid):
    name_list = []
    #localdb = "/teraraid3/pemsley/emdb/structures/"
    localdb = "/cephfs/ranganaw/EMDB/"
    xmlfile1 = "EMD-%s/header/emd-%s.xml" % (emdid, emdid)
    tree1 = ET.parse(localdb + xmlfile1)
    root1 = tree1.getroot()
    root = root1
    path = "./"
    model = None
    maskid = None
    half1id = None
    half2id = None
    claimed_resol = None
    pointg = None

    for crossreferences in root.findall("crossreferences"):
        for pdb_list in crossreferences.findall("pdb_list"):
            for pdb_reference in pdb_list.findall("pdb_reference"):
                for pdb_id in pdb_reference.findall("pdb_id"):
                    model = pdb_id.text

    for structure_determination_list in root.findall("structure_determination_list"):
        for structure_determination in structure_determination_list.findall("structure_determination"):
            for singleparticle_processing in structure_determination.findall("singleparticle_processing"):
                for final_reconstruction in singleparticle_processing.findall("final_reconstruction"):
                    for resolution in final_reconstruction.findall("resolution"):
                        claimed_resol = resolution.text
                    for symmetry in final_reconstruction.findall("applied_symmetry"):
                        for pointgroup in symmetry.findall("point_group"):
                            pointg = pointgroup.text

    for interpretation in root.findall("interpretation"):
        for segmentation_list in interpretation.findall("segmentation_list"):
            for segmentation in segmentation_list.findall("segmentation"):
                for file in segmentation.findall("file"):
                    maskid = file.text

        for half_map_list in interpretation.findall("half_map_list"):
            for i, half_map in enumerate(half_map_list.findall("half_map")):
                if i == 0:
                    for file in half_map.findall("file"):
                        half1id = file.text
                if i == 1:
                    for file in half_map.findall("file"):
                        half2id = file.text

    for map in root.findall("map"):
        for file in map.findall("file"):
            mapid = file.text

    print("claimed resol ", claimed_resol)
    print("Mask Id: ", maskid)
    print("Half1id: ", half1id)
    print("Half2id: ", half2id)

    # check if mask file presents
    maskname = "emd_%s_msk_1.map"%(emdid)
    maskpath = localdb+"EMD-%s/masks/emd_%s_msk_1.map"%(emdid, emdid)
    ismask = os.path.isfile(maskpath)

    if claimed_resol is None or float(claimed_resol) >= 10.0:
        return []
    if pointg is None:
        return []
    if ismask:
        readname_list = []
        writename_list = []
        if half1id is not None:
            # copy mask here
            shutil.copy2(maskpath, path + maskname)
            readname_list.append("EMD-%s/other/%s" % (emdid, half1id))
            writename_list.append("emd_%s_half_map_1.map" % (emdid))
            readname_list.append("EMD-%s/other/%s" % (emdid, half2id))
            writename_list.append("emd_%s_half_map_2.map" % (emdid))
        try:
            for readname, writename in zip(readname_list, writename_list):
                name_list.append(path + writename)
                if readname.endswith((".map")):
                    shutil.copy2(localdb + readname, path + writename)
                elif readname.endswith((".gz")):
                    with gzip.open(localdb + readname, "rb") as fmap:
                        file_content = fmap.read()
                    with open(path + writename, "wb") as f:
                        f.write(file_content)
            return [name_list, claimed_resol, pointg, maskname]
        except Exception as e:
            raise e


if __name__=="__main__":
    results = fetch_halfmaps(23356)
    print(results)