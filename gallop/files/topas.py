# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for working with TOPAS files.
"""

import numpy as np
import pymatgen as pmg

def get_TOPAS_matrix(lines):
    """
    Read some lines from a TOPAS output file that correspond to
    a Covariance matrix and return a numpy array of the matrix

    Args:
        lines (List): List of the lines in a TOPAS output file

    Returns:
        numpy array: The topas covariance matrix
    """
    matrix = []
    for line in lines:
        temp = []
        for item in line[2:]:
            if "iprm" in line[0]:
                try:
                    temp.append(float(item))
                except ValueError:
                    minus_idx = [i for i, ltr in enumerate(item) if ltr == "-"]
                    split = item.split("-")
                    if len(minus_idx) > 1:
                        temp.append(-1*float(split[1]))
                        temp.append(-1*float(split[2]))
                    else:
                        temp.append(float(split[0]))
                        temp.append(-1*float(split[1]))
        matrix.append(temp)
    return np.array(matrix)

def get_data_from_TOPAS_output(filename, percentage_cutoff_inv_cov=20):
    """
    Note: This assumes that the only refined parameters in the Pawley refinement
    are the peak intensities.
    Args:
        filename ([type]): [description]
        percentage_cutoff_inv_cov (int, optional): [description]. Defaults to 20

    Returns:
        dict: A dictionary of the required information
    """

    lines = []
    with open(filename, "r") as in_file:
        for line in in_file:
            line = list(filter(None,line.strip().split(" ")))
            if len(line) > 0:
                lines.append(line)
    in_file.close()

    cell = {}
    hkl = []
    m = []
    dspacing = []
    twotheta = []
    I = []
    #sigma = []
    C_matrix = []
    gof = 1
    for i, line in enumerate(lines):
        if "la" in line and "1" in line and "lo" in line:
            wavelength = line[-1]
        if len(line) == 2:
            if "a" in line:
                cell["a"] = float(line[1].split("_")[0].strip("`"))
            if "b" in line:
                cell["b"] = float(line[1].split("_")[0].strip("`"))
            if "c" in line:
                cell["c"] = float(line[1].split("_")[0].strip("`"))
            if "al" in line:
                cell["al"] = float(line[1].split("_")[0].strip("`"))
            if "be" in line:
                cell["be"] = float(line[1].split("_")[0].strip("`"))
            if "ga" in line:
                cell["ga"] = float(line[1].split("_")[0].strip("`"))
        if "space_group" in line:
            cell["space_group"] = " ".join(line[1:]).strip("\"")
        if "gof" in line:
            gof *= float(line[1])
        if len(line) == 8 and "@" in line:
            hkl.append([int(x) for x in line[0:3]])
            m.append(int(line[3]))
            dspacing.append(line[4])
            twotheta.append(line[5])
            I_sig = line[-1].split("`_")
            I.append(float(I_sig[0]))
            #sigma.append(float(I_sig[1]))
        else:
            if "C_matrix" in line:
                n_peaks = int(lines[i+2][-1])
                C_matrix = get_TOPAS_matrix(lines[i+3:i+2+n_peaks+1])

    hkl = np.array(hkl)
    m = np.array(m)
    dspacing = np.array(dspacing)
    twotheta = np.array(twotheta)
    I = np.array(I)
    I_mult_corrected = I/m
    C_matrix_mult_corrected = np.diag(1/m) @ C_matrix @ np.diag(1/m)
    #npeaks = len(hkl)
    #while not np.all(
    #    np.linalg.eigvals(
    #    np.linalg.inv(C_matrix_mult_corrected[:npeaks][:,:npeaks])) > 0):
    #    npeaks -= 1
    #if npeaks < len(hkl):
    #    print("WARNING - inverse covariance matrix is not positive definite.")
    #    print("Recommend reducing peak count to",npeaks)
    #    print("This is",round(100*float(npeaks)/len(hkl),2),"% of the peaks "
    #            "in the Pawley file")

    ############################################################################
    # Still need to work out how to implement the % correlation cutoff
    # It's proving difficult because the matrices from TOPAS I've got give
    # negative values on the inverse matrix diagonal which prevents me from
    # converting to the inverse correlation matrix.
    ############################################################################
    data = {}
    data["percentage_cutoff_inv_cov"] = percentage_cutoff_inv_cov
    data["hkl"] = hkl.astype(int)
    data["intensities"] = I_mult_corrected
    data["inverse_covariance_matrix"] = np.linalg.inv(C_matrix_mult_corrected)
    data["lattice"] = pmg.core.Lattice.from_parameters(cell["a"],
                                                    cell["b"],
                                                    cell["c"],
                                                    cell["al"],
                                                    cell["be"],
                                                    cell["ga"])
    space_group = pmg.symmetry.groups.SpaceGroup(cell["space_group"])

    data["sg_number"] = int(space_group.int_number)
    data["original_sg_number"] = data["sg_number"]
    data["space_group"] = space_group
    data["dspacing"] = dspacing
    data["twotheta"] = twotheta
    data["wavelength"] = wavelength
    return data