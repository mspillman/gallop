# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for working with DASH files.
"""

import os
import numpy as np
import pymatgen as pmg
from pymatgen.symmetry import groups


def get_data_from_DASH_sdi(filename, percentage_cutoff_inv_cov=20):
    """
    Read in a DASH .sdi file to obtain numpy arrays of:
        - hkls
        - twotheta for hkls
        - peak intensities
        - inverse covariance matrix
    As well as:
        - pymatgen lattice
        - pymatgen Space group

    Args:
        filename (str): The filename of the .sdi to read in. The associated
            hcv, dsl and tic files should also be in the same directory and
            accessible.
        percentage_cutoff (int, optional):  the minimum percentage correlation
                        to be included in the inverse covariance matrix.
                        Defaults to 20 to be comparable with DASH, however, this
                        doesn't affect the speed of GALLOP on GPU so can be
                        freely set without impacting performance.
    Returns:
        dict: A dictionary with the required information
    """
    data = {}
    data["hcv"], data["tic"], data["dsl"], data["unit_cell"], \
                data["sg_number"], data["sg_setting"] = read_DASH_sdi(filename)
    data["wavelength"] = read_DASH_dsl(data["dsl"])
    data["original_sg_number"] = data["sg_number"]
    data["hkl"], data["intensities"], data["sigma"],\
        data["inverse_covariance_matrix"] = read_DASH_hcv(
            data["hcv"], percentage_cutoff_inv_cov=percentage_cutoff_inv_cov)
    data["twotheta"] = read_DASH_tic(data["tic"])
    data["lattice"] = pmg.Lattice.from_parameters(data["unit_cell"][0],
                                                    data["unit_cell"][1],
                                                    data["unit_cell"][2],
                                                    data["unit_cell"][3],
                                                    data["unit_cell"][4],
                                                    data["unit_cell"][5])
    data["space_group"] = groups.SpaceGroup.from_int_number(data["sg_number"])
    return data


def get_DASH_inverse_covariance_matrix(
                    off_diag_elements, sigma, percentage_cutoff_inv_cov=20):
    """
    Read in a DASH-produced list of correlated peaks and produce
    a 2D numpy inverse covariance matrix.
        -   Populate the diagonals with sigma^2
        -   Populate the off-diagonals as raw decimals rather than
            percentages to save having to do it later.

    Args:
        off_diag_elements (numpy array): Non-zero off-diagonal elements
            of the inverse correlation matrix stored in the hcv, which
            will be converted into the covariance matrix
        sigma (numpy array): The square-root of the diagonal elements of
            the inverse covariance matrix
        percentage_cutoff_inv_cov (int, optional): the minimum percentage
            correlation to be included in the inverse covariance matrix.
            Defaults to 20 to be comparable with DASH, however, this doesn't
            affect the speed of GALLOP so can be freely set without impacting
            performance.

    Returns:
        numpy array: The inverse covariance matrix
    """
    matrix = np.zeros((len(off_diag_elements), len(off_diag_elements)))
    for i in range(0,len(off_diag_elements)):
        if i != len(off_diag_elements)-1:
            j = i+1
            # Populate diagonals
            matrix[i][i] += sigma[i]**2
            # Populate off diagonal elements
            for corr in off_diag_elements[i]:
                if float(corr) >= percentage_cutoff_inv_cov:
                    matrix[i][j] = 0.01*sigma[i]*sigma[j]*float(corr)
                    matrix[j][i] += matrix[i][j]
                j+=1
        else:
            matrix[i][i] += sigma[i]**2
    return matrix


def read_DASH_hcv(filename, percentage_cutoff_inv_cov=20):
    """
    Read in a DASH-produced hcv, and return numpy arrays of hkl,
    intensity, sigma values together with a 2D numpy inverse_covariance
    matrix.
    Format of DASH hcv files:
    h, k, l, I, sigma, count, off-diag inv. of correlation matrix
    See function get_DASH_inverse_covariance_matrix for how the
    inverse_covariance_matrix is populated using the sigma and
    inverse correlation matrix values.

    Args:
        filename (str): Name for the DASH .sdi file. File is assumed
            to be in the working directory
        percentage_cutoff_inv_cov (int, optional): the minimum percentage
            correlation to be included in the inverse covariance
            matrix. Defaults to 20 to be comparable with DASH,
            however, this doesn't affect the speed of GALLOP so can
            be set as desired without impacting performance.

    Returns:
        tuple:  Numpy arrays of hkl, intensity, sigma and the inverse
                covariance matrix
    """

    hkl = []
    I = []
    sigma = []
    inverse_covariance_off_diag = []
    peaknumbers = []
    with open(filename) as in_hcv:
        for line in in_hcv:
            line = list(filter(None,line.strip().split(" ")))
            hkl.append(line[0:3])
            I.append(line[3])
            sigma.append(line[4])
            peaknumbers.append(int(line[5]))
            inverse_covariance_off_diag.append(line[6:])
    in_hcv.close()
    hkl = np.array(hkl).astype(int)
    I = np.array(I).astype(float)
    sigma = np.array(sigma).astype(float)
    peaknumbers = np.array(peaknumbers)
    inverse_covariance = get_DASH_inverse_covariance_matrix(
                                inverse_covariance_off_diag,
                                sigma,
                                percentage_cutoff_inv_cov=percentage_cutoff_inv_cov
                                )
    return hkl, I, sigma, inverse_covariance

def read_DASH_dsl(filename):
    """
    Read a DASH dsl file to obtain the wavelength used

    Args:
        filename (str): Filename of the dsl file

    Returns:
        float: the wavelength used
    """
    with open(filename) as in_dsl:
        for line in in_dsl:
            line = list(filter(None,line.strip().split(" ")))
            if "rad" in line:
                return float(line[1])

def read_DASH_tic(filename):
    """
    Read in a DASH-produced tic file, and return numpy array of
    twotheta values

    Args:
        filename (str): Filename of the .tic file

    Returns:
        Numpy array: the two-theta values for each hkl
    """
    twotheta = []
    with open(filename) as in_tic:
        for line in in_tic:
            line = list(filter(None,line.strip().split(" ")))
            twotheta.append(line[3])
    in_tic.close()
    twotheta = np.array(twotheta).astype(float)
    return twotheta

def read_DASH_sdi(filename):
    """
    Read a DASH .sdi file, and obtain the filenames for the the .hcv
    and .tic, as well as the unit cell lattice parameters and the
    space group.

    Args:
        filename (str): Filename of the DASH .sdi file

    Returns:
        tuple: contains the filenames of hcv, tic files as well as
            the unit cell, spacegroup and its setting.
    """
    directory = os.path.split(filename)[0]
    with open(filename) as in_sdi:
        for line in in_sdi:
            line = list(filter(None,line.strip().split(" ")))
            if line[0] == "HCV":
                hcv = os.path.join(os.getcwd(),directory,line[1].strip(".\\"))
            if line[0] == "TIC":
                tic = os.path.join(os.getcwd(),directory,line[1].strip(".\\"))
            if line[0] == "DSL":
                dsl = os.path.join(os.getcwd(),directory,line[1].strip(".\\"))
            if line[0] == "Cell":
                unit_cell = np.array(line[1:]).astype(float)
            if line[0] == "SpaceGroup":
                sg = int(line[2].split(":")[0])
                sg_setting = line[2].split(":")[-1]
    in_sdi.close()
    return hcv, tic, dsl, unit_cell, sg, sg_setting