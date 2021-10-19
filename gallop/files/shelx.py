# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for working with SHELX single crystal files.

Data are assumed to be in HKLF 4 SHELX format
"""

import numpy as np
import pymatgen as pmg
from pymatgen.symmetry import groups

def get_shelx_data(filename, hklfile):
    """
    Read in SX files in SHELX hkl & ins format to obtain numpy arrays of:
        - hkls
        - peak intensities
        - peak esds
    As well as:
        - pymatgen lattice
        - pymatgen Space group

    Args:
        insfile (str): Filename for *.ins containing unit cell and SG info
        hklfile (str): Filename for *.hkl containing peak info in HKLF 4 format

    Returns:
        dict: A dictionary with the required information
    """
    if hklfile is None:
        hklfile = filename.rsplit(".", 1)[0]+".hkl"
    data = {}
    fileformat = filename.rsplit(".", 1)[1]
    if "cif" in fileformat:
        space_group_number, cell, wavelength = get_data_from_cif(filename)
        data["space_group"] = groups.SpaceGroup.from_int_number(space_group_number)
    elif "ins" in fileformat:
        space_group, cell, wavelength = get_data_from_ins(filename)
        data["space_group"] = groups.SpaceGroup(space_group)
    else:
        raise ValueError("Unknown format file format for",filename,format)

    hkl, I, sigma = get_data_from_hkl(hklfile)

    data["lattice"] = pmg.Lattice.from_parameters(cell[0], cell[1], cell[2],
                                                    cell[3], cell[4], cell[5])

    data["sg_number"] = int(data["space_group"].int_number)
    data["original_sg_number"] = data["sg_number"]
    data["wavelength"] = wavelength


    d_hkl = 1/np.sqrt(np.sum(np.dot(hkl,
        data["lattice"].reciprocal_lattice_crystallographic.matrix)**2, axis=1))
    d_hkl_sort = np.argsort(-1*d_hkl) # Get descending order for d-spacing

    data["hkl"] = hkl[d_hkl_sort]
    data["intensities"] = I[d_hkl_sort]
    data["dspacing"] = d_hkl[d_hkl_sort]

    cov = np.diag(sigma) @ np.eye(sigma.shape[0]) @ np.diag(sigma)
    inv_cov = np.linalg.inv(cov)
    data["inverse_covariance_matrix"] = inv_cov
    return data


def get_data_from_ins(filename):
    wavelength = None
    with open(filename) as insfile:
        for line in insfile:
            line = list(filter(None,line.strip().split(" ")))
            if len(line) > 0:
                if line[0] == "TITL":
                    space_group = line[-1]
                if line[0] == "CELL":
                    cell = np.array(line[1:]).astype(float)
                    wavelength = float(line[1])

    return space_group, cell, wavelength

def get_data_from_cif(filename):
    wavelength = None
    with open(filename) as cif:
        for line in cif:
            line = list(filter(None,line.strip().split(" ")))
            if len(line) > 0:
                if "length_a" in line[0]:
                    a = float(line[1].split("(")[0])
                if "length_b" in line[0]:
                    b = float(line[1].split("(")[0])
                if "length_c" in line[0]:
                    c = float(line[1].split("(")[0])
                if "angle_alpha" in line[0]:
                    al = float(line[1].split("(")[0])
                if "angle_beta" in line[0]:
                    be = float(line[1].split("(")[0])
                if "angle_gamma" in line[0]:
                    ga = float(line[1].split("(")[0])
                if "Int_Tables_number" in line[0]:
                    space_group_number = int(line[1])
                if "wavelength.wavelength" in line[0] or "wavelength_wavelength" in line[0]:
                    wavelength = float(line[1])

    cif.close()
    cell = np.array([a,b,c,al,be,ga])
    return space_group_number, cell, wavelength


def get_data_from_hkl(filename):
    hkl = []
    I = []
    sigma = []
    with open(filename) as hklfile:
        for line in hklfile:
            line = list(filter(None,line.strip().split(" ")))
            if len(line) > 0:
                hkl.append(line[0:3])
                I.append(line[3])
                sigma.append(line[4])
    hklfile.close()
    hkl = np.array(hkl).astype(int)
    I = np.array(I).astype(float)
    sigma = np.array(sigma).astype(float)
    return hkl, I, sigma