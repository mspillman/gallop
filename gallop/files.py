# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for working with files.
"""

import numpy as np
import pymatgen as pmg
from pymatgen.symmetry import groups
from pymatgen.io import cif
from pymatgen.core.operations import SymmOp
from pymatgen.io.cif import CifBlock, CifFile
from collections import OrderedDict
from monty.io import zopen
import time
import os
import pickle
import glob
import py3Dmol

import gallop.zm_to_cart as zm_to_cart

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


def get_data_from_GSAS_gpx(filename, percentage_cutoff_inv_cov=20):
    """
    Read in a GSAS project file (.gpx) to obtain numpy arrays of:
        - hkls
        - peak intensities
        - inverse covariance matrix
    As well as:
        - pymatgen lattice
        - pymatgen Space group

    Args:
        filename (str): [description]
        percentage_cutoff_inv_cov (int, optional):  the minimum percentage
                        correlation to be included in the inverse
                        covariance matrix. Defaults to 20 to be
                        comparable with DASH, however, this doesn't
                        affect the speed of GALLOP so can be freely
                        set without impacting performance.
    Returns:
        dict: A dictionary with the required information
    """
    data, names = IndexGPX(filename)

    instrument_params = data[names[-2][0]][names[-2][4]][0]
    if "Lam" in instrument_params.keys():
        wavelength = instrument_params["Lam"][1]
    elif "Lam1" in instrument_params.keys():
        wavelength = instrument_params["Lam1"][1]

    space_group = data["Phases"][names[-1][-1]]["General"]["SGData"]["SpGrp"]
    cell = data["Phases"][names[-1][-1]]["General"]["Cell"][1:-1]

    hkl, dspacing, I, sigma = [], [], [], []
    for d in data["Phases"][names[-1][-1]]["Pawley ref"]:
        hkl.append(d[:3])
        dspacing.append(d[4])
        I.append(d[-2])
        sigma.append(d[-1])

    hkl = np.array(hkl).astype(int)
    dspacing = np.array(dspacing)
    I = np.array(I)
    sigma = np.array(sigma)

    names_subset = data["Covariance"]["data"]["varyList"]
    names_all = data["Covariance"]["data"]["varyListStart"]
    cov_subset = data["Covariance"]["data"]["covMatrix"]

    intensities, inverse_covariance_matrix = get_GSAS_inv_cov_matrix(cov_subset,
                            names_all, names_subset, I,
                            percentage_cutoff_inv_cov=percentage_cutoff_inv_cov)
    data = {}
    data["wavelength"] = wavelength
    data["hkl"] = hkl
    data["intensities"] = intensities
    data["dspacing"] = dspacing
    data["inverse_covariance_matrix"] = inverse_covariance_matrix
    data["lattice"] = pmg.Lattice.from_parameters(cell[0], cell[1], cell[2],
                                                    cell[3], cell[4], cell[5])
    data["space_group"] = pmg.symmetry.groups.SpaceGroup(space_group)
    data["sg_number"] = int(data["space_group"].int_number)
    data["original_sg_number"] = data["sg_number"]
    return data

def IndexGPX(GPXfile):
    '''
    Pinched from the GSAS-II source code, and modified to allow it to work
    without needing extra GSAS related imports. See original source here:
    https://gsas-ii.readthedocs.io/en/latest/_modules/GSASIIstrIO.html#IndexGPX

    Original docstring below:
    >    Create an index to a GPX file, optionally the file into memory.
    >    The byte size of the GPX file is saved. If this routine is called
    >    again, and if this size does not change, indexing is not repeated
    >    since it is assumed the file has not changed (this can be overriden
    >    by setting read=True).
    >
    >    :param str GPXfile: full .gpx file name
    >    :returns: Project,nameList if read=, where
    >
    >     * Project (dict) is a representation of gpx file following the GSAS-II
    >       tree structure for each item: key = tree name (e.g. 'Controls',
    >       'Restraints', etc.), data is dict
    >     * nameList (list) has names of main tree entries & subentries used to
    >       reconstruct project file
    '''
    gpxIndex = {}
    gpxNamelist = []
    fp = open(GPXfile,'rb')
    Project = {}
    try:
        while True:
            pos = fp.tell()
            data = pickle.load(fp)
            datum = data[0]
            gpxIndex[datum[0]] = pos
            Project[datum[0]] = {'data':datum[1]}
            gpxNamelist.append([datum[0],])
            for datus in data[1:]:
                Project[datum[0]][datus[0]] = datus[1]
                gpxNamelist[-1].append(datus[0])
    except EOFError:
        pass
    fp.close()
    return Project, gpxNamelist

def get_GSAS_inv_cov_matrix(cov_subset, names_all, names_subset, I,
                                                percentage_cutoff_inv_cov=20):
    """
    Inverts the covariance matrix from GSAS, and use this to rebuild a full
    inverse covariance matrix that includes the hkls that were excluded in
    the GSAS-II Pawley fitting procedure (due to overlap/equivalence etc)

    Args:
        cov_subset (np.array): The covariance matrix from GSAS. This will be of
                        shape names_subset x names_subset
        names_all (list): A list of all of the peaks within the data range used
                        for the Pawley fit in GSAS
        names_subset (list): The non-equivalent peaks used by GSAS to perform
                        the Pawley refinement
        percentage_cutoff_inv_cov (int, optional):  the minimum percentage
                        correlation to be included in the inverse
                        covariance matrix. Defaults to 20 to be comparable to
                        DASH. There is no difference in speed between 0 and 20
                        (or any other number from 0-100) so can be tested freely
                        without impacting run times.

    Returns:
        np.array: The modified intensity for all hkls
        np.array: The full inverse covariance matrix for all hkls
    """

    # Find the peaks not in the Pawley refinement and account for any other
    # refined variables, as we only want the peak-to-peak covariance
    peak_indices = [i for i, x in enumerate(names_subset) if "PWLref" in x]

    names_all = [x for x in names_all if "PWL" in x]
    names_subset = [x for x in names_subset if "PWL" in x]
    missing_peaks = [x for x in names_all if x not in names_subset]

    # Invert the covariance matrix and convert to the inverse correlation matrix
    inv_cov_subset = np.linalg.inv(cov_subset)
    # Select only the peak-to-peak inverse covariances
    inv_cov_subset = inv_cov_subset[peak_indices][:,peak_indices]
    inv_sigma_subset = np.sqrt(np.diag(inv_cov_subset))
    inv_cor_subset = (np.diag(1/inv_sigma_subset)
                        @ inv_cov_subset
                        @ np.diag(1/inv_sigma_subset))

    # Now we have the inverse correlation rather than covariance matrix,
    # we can build it up to include all of the peaks, then rebuild the inverse
    # covariance matrix from that. We also need to modify the intensities and
    # 1/sigmas of the peaks that were excluded from the Pawley fit.
    # Effectively, this is converting the output to look like a DASH hcv, from
    # which the same logic is then used to rebuild the inverse covariance matrix
    inv_cor_full = []
    inv_sigma_full = []
    I_mod = np.copy(I)
    j=0
    for i, x in enumerate(names_all):
        if x not in missing_peaks:
            correlations = inv_cor_subset[i-j][i-j+1:]
            inv_cor_full.append(np.around(correlations, 2).tolist())
            inv_sigma_full.append(inv_sigma_subset[i-j])
        else:
            inv_sigma_full.append(None)
            inv_cor_full.append(None)
            j+=1

    for i, s in enumerate(inv_sigma_full):
        if s is None:
            inv_sigma_full[i] = inv_sigma_full[i-1]

    inv_sigma_mod = np.copy(inv_sigma_full)
    for i, c in enumerate(inv_cor_full):
        if c is not None:
            k = 0
            if i < (len(inv_cor_full)-1):
                while inv_cor_full[i+k+1] is None:
                    if i + k + 2 == len(inv_cor_full):
                        break
                    k+=1
                if k > 0:
                    base_correlations = inv_cor_full[i]
                    for j in range(k+1):
                        inv_cor_full[i+j] = (k-j)*[1.] + base_correlations
                        # Divide the intensity of the overlapped peaks by the
                        # number of peaks that were overlapped
                        I_mod[i+j] = I_mod[i+j] / (k+1.)
                        # Multiply the inverse sigma of the overlapped peaks by
                        # the number of peaks that were overlapped
                        inv_sigma_mod[i+j] = inv_sigma_mod[i+j] * (k+1.)

    # Now reconstruct the full inverse correlation matrix
    inv_cor_matrix = np.zeros((len(inv_cor_full), len(inv_cor_full)))
    for i in range(0,len(inv_cor_full)):
        if i != len(inv_cor_full)-1:
            j = i+1
            # Populate diagonals
            inv_cor_matrix[i][i] = 1.
            # Populate off diagonal elements
            for corr in inv_cor_full[i]:
                if np.abs(corr) >= (percentage_cutoff_inv_cov/100.):
                    inv_cor_matrix[i][j] = float(corr)
                    inv_cor_matrix[j][i] = inv_cor_matrix[i][j]
                j+=1
        else:
            inv_cor_matrix[i][i] = 1.

    # Finally, get the inverse covariance matrix needed for chi2 calcs.
    inverse_covariance_matrix = (np.diag(inv_sigma_mod)
                                @ inv_cor_matrix
                                @ np.diag(inv_sigma_mod))
    return I_mod, inverse_covariance_matrix

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
    data["lattice"] = pmg.Lattice.from_parameters(cell["a"],
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


def get_CIF_atomic_coords_and_species(Structure, filename,
                                        add_CIF_dw_factors=True):
    """
    Read a CIF and extract the fractional atomic-coordinates.
    It is assumed that a GALLOP Structure object has been created
    that already contains the correct unit cell and PXRD Pawley data.

    These coordinates are added to the Structure object.
    Args:
        Structure (class): GALLOP structure object
        filename (str): Filename of the CIF. Assumed to be in the working
            directory.
        add_CIF_dw_factors (bool, optional): Include Debye-Waller factors
            for the atoms in the CIF. By default, non-H = 3, H = 6.
            Defaults to True.
    """

    print("Ensure that CIF and structure fit files are in the same setting")
    cif_file_structure = cif.CifParser(filename).as_dict()
    cif_file_structure = cif_file_structure[list(cif_file_structure.keys())[0]]
    Structure.cif_species = cif_file_structure["_atom_site_type_symbol"]
    x = cif_file_structure["_atom_site_fract_x"]
    y = cif_file_structure["_atom_site_fract_y"]
    z = cif_file_structure["_atom_site_fract_z"]

    x = np.array([v.split("(")[0] for v in x]).astype(float)
    y = np.array([v.split("(")[0] for v in y]).astype(float)
    z = np.array([v.split("(")[0] for v in z]).astype(float)
    Structure.cif_frac_coords = np.vstack([x,y,z]).T
    if add_CIF_dw_factors:
        elements = list(set(Structure.cif_species))
        Structure.CIF_dw_factors = {}
        for e in elements:
            if e != "H":
                Structure.CIF_dw_factors[e] = 3.
            else:
                Structure.CIF_dw_factors[e] = 6.
    no_H_coords = []
    no_H_species = []
    if Structure.ignore_H_atoms:
        for i, e in enumerate(cif_file_structure["_atom_site_type_symbol"]):
            if e != "H":
                no_H_species.append(e)
                no_H_coords.append(Structure.cif_frac_coords[i])
        Structure.cif_species_no_H = no_H_species
        Structure.cif_frac_coords_no_H = np.vstack(no_H_coords)

class DASHCifWriter:
    """
    A modified wrapper around the pymatgen CifFile to write CIFs from pymatgen
    structures.

    See "CifWriter" on this page for the original code:
    https://pymatgen.org/pymatgen.io.cif.html

    Modification is to stop the unit cell (and hence coordinates) from being
    "standardised". This means that the output CIF is in the same setting as
    the input data which allows for easier comparison etc.
    """
    def __init__(self, struct, symprec=None, significant_figures=8, sg_number=1,
                comment=None, site_labels=None):
        """
        Args:
            struct (Structure): structure to write
            symprec (float): If not none, finds the symmetry of the structure
                and writes the cif with symmetry information. Passes symprec
                to the SpacegroupAnalyzer.
            significant_figures (int): Specifies precision for formatting of
                floats. Defaults to 8.
            angle_tolerance (float): Angle tolerance for symmetry finding.
                Passes angle_tolerance to the SpacegroupAnalyzer.
                Used only if symprec is not None.
        """

        format_str = "{:.%df}" % significant_figures

        block = OrderedDict()
        loops = []

        spacegroup = (pmg.symmetry.groups.sg_symbol_from_int_number(sg_number),
                        str(sg_number))

        latt = struct.lattice
        comp = struct.composition
        no_oxi_comp = comp.element_composition
        block["_symmetry_space_group_name_H-M"] = spacegroup[0]
        for cell_attr in ['a', 'b', 'c']:
            block["_cell_length_" + cell_attr] = format_str.format(
                getattr(latt, cell_attr))
        for cell_attr in ['alpha', 'beta', 'gamma']:
            block["_cell_angle_" + cell_attr] = format_str.format(
                getattr(latt, cell_attr))
        block["_symmetry_Int_Tables_number"] = spacegroup[1]
        block["_chemical_formula_structural"] = no_oxi_comp.reduced_formula
        block["_chemical_formula_sum"] = no_oxi_comp.formula
        block["_cell_volume"] = format_str.format(latt.volume)

        _, fu = no_oxi_comp.get_reduced_composition_and_factor()
        block["_cell_formula_units_Z"] = str(int(fu))

        if symprec is None:
            block["_symmetry_equiv_pos_site_id"] = ["1"]
            block["_symmetry_equiv_pos_as_xyz"] = ["x, y, z"]
        else:
            symmops = []
            sg_symbol = pmg.symmetry.groups.sg_symbol_from_int_number(sg_number)
            SG = pmg.symmetry.groups.SpaceGroup(sg_symbol)
            for op in SG.symmetry_ops:
                v = op.translation_vector
                symmops.append(SymmOp.from_rotation_and_translation(
                    op.rotation_matrix, v))

            ops = [op.as_xyz_string() for op in symmops]
            block["_symmetry_equiv_pos_site_id"] = \
                ["%d" % i for i in range(1, len(ops) + 1)]
            block["_symmetry_equiv_pos_as_xyz"] = ops

        loops.append(["_symmetry_equiv_pos_site_id",
                    "_symmetry_equiv_pos_as_xyz"])

        try:
            symbol_to_oxinum = OrderedDict([
                (el.__str__(),
                float(el.oxi_state))
                for el in sorted(comp.elements)])
            block["_atom_type_symbol"] = symbol_to_oxinum.keys()
            block["_atom_type_oxidation_number"] = symbol_to_oxinum.values()
            loops.append(["_atom_type_symbol",
                            "_atom_type_oxidation_number"])
        except (TypeError, AttributeError):
            symbol_to_oxinum = OrderedDict([(el.symbol, 0) for el in
                                            sorted(comp.elements)])

        atom_site_type_symbol = []
        atom_site_symmetry_multiplicity = []
        atom_site_fract_x = []
        atom_site_fract_y = []
        atom_site_fract_z = []
        atom_site_label = []
        atom_site_occupancy = []
        count = 0
        for site in struct:
            for sp, occu in sorted(site.species.items()):
                atom_site_type_symbol.append(sp.__str__())
                atom_site_symmetry_multiplicity.append("1")
                atom_site_fract_x.append(format_str.format(site.a))
                atom_site_fract_y.append(format_str.format(site.b))
                atom_site_fract_z.append(format_str.format(site.c))
                atom_site_label.append("{}{}".format(sp.symbol, count))
                atom_site_occupancy.append(occu.__str__())
                count += 1

        block["_atom_site_type_symbol"] = atom_site_type_symbol
        if site_labels is not None:
            block["_atom_site_label"] = site_labels
        else:
            block["_atom_site_label"] = atom_site_label
        block["_atom_site_symmetry_multiplicity"] = \
                                                atom_site_symmetry_multiplicity
        block["_atom_site_fract_x"] = atom_site_fract_x
        block["_atom_site_fract_y"] = atom_site_fract_y
        block["_atom_site_fract_z"] = atom_site_fract_z
        block["_atom_site_occupancy"] = atom_site_occupancy
        loops.append(["_atom_site_type_symbol",
                    "_atom_site_label",
                    "_atom_site_symmetry_multiplicity",
                    "_atom_site_fract_x",
                    "_atom_site_fract_y",
                    "_atom_site_fract_z",
                    "_atom_site_occupancy"])

        d = OrderedDict()
        d[comp.reduced_formula] = CifBlock(block, loops, comp.reduced_formula)
        comment_for_file = "# Generated using pymatgen and GALLOP"
        if comment is not None:
            comment_for_file += "\n" + comment
        self.cf = CifFile(d, comment=comment_for_file)

    @property
    def ciffile(self):
        """
        Returns: CifFile associated with the CifWriter.
        """
        return self.cf

    def __str__(self):
        """
        Returns the cif as a string.
        """
        return self.cf.__str__()

    def write_file(self, filename):
        """
        Write the cif file.
        """
        with zopen(filename, "wt") as f:
            f.write(self.__str__())

def save_CIF_of_best_result(Structure, result, start_time=None,
    n_reflections=None, filename_root=None):
    """
    Save a CIF of the best result obtained in a minimise run.
    Filename contains information about the run. For example, if the result
    is the first run, and the best chi_2 value is 50.1, 300 reflections were
    used in the chi_2 calculation and the time taken to obtain this solution was
    1.2 minutes, then the filename would be:

        filename_root_001_50.1_chisqd_300_refs_1.2_mins.cif

    Args:
        Structure (class): GALLOP structure object
        result (dict): Dictionary of results obtained by minimise
        start_time (time): time.time() when the runs began
        n_reflections (int, optional): the number of reflections used for chi_2
        calc. Defaults to None.
        filename_root (str, optional): If None, takes the name from the
            Structure. Root of the filename can be overwritten with this
            argument. Defaults to None.
    """
    external = result["external"]
    internal = result["internal"]
    chi_2 = result["chi_2"]
    run = result["GALLOP Iter"] + 1
    if n_reflections is None:
        n_reflections = len(Structure.hkl)
    elif n_reflections > len(Structure.hkl):
        n_reflections = len(Structure.hkl)
    # For the purpose of outputting a CIF, include the H-atoms back in. However,
    # the GALLOP runs may not be finished, so save whatever the parameter is set
    # to, then restore this setting once the CIF has been written.
    ignore_H_setting = Structure.ignore_H_atoms

    Structure.ignore_H_atoms = False

    # Save a CIF of the best particle found
    best_frac_coords = zm_to_cart.get_asymmetric_coords_from_numpy(Structure,
            external[chi_2 == chi_2.min()], internal[chi_2 == chi_2.min()],
            verbose=False)

    species = []
    for zm in Structure.zmatrices:
        species += zm.elements

    all_atom_names = []
    for zmat in Structure.zmatrices:
        all_atom_names += zmat.atom_names
    if len(set(all_atom_names)) == len(all_atom_names):
        site_labels = all_atom_names
    else:
        site_labels = None

    output_structure = pmg.Structure(lattice=Structure.lattice, species=species,
                                    coords=best_frac_coords[0][:len(species)])
    ext_comment = "# GALLOP External Coords = " + ",".join(list(
                                external[chi_2 == chi_2.min()][0].astype(str)))
    int_comment = "# GALLOP Internal Coords = " + ",".join(list(
                                internal[chi_2 == chi_2.min()][0].astype(str)))
    comment = ext_comment + "\n" + int_comment
    writer = DASHCifWriter(output_structure, symprec=1e-12,
                            sg_number=Structure.original_sg_number,
                            comment=comment, site_labels=site_labels)

    if filename_root is None:
        filename_root = Structure.name
    if start_time is not None:
        filename = (filename_root
                + "_{:04d}_{:.3f}_chisqd_{}_refs_{:.1f}_mins.cif")
        filename = filename.format(run, chi_2.min(), n_reflections,
                            (time.time()-start_time)/60)
    else:
        filename = (filename_root
                + "_{:04d}_{:.3f}_chisqd_{}_refs.cif")
        filename = filename.format(run, chi_2.min(), n_reflections)
    ciffile = writer.cf.data[list(writer.cf.data.keys())[0]]
    # Add the filename to the data_ string in the CIF to make it easier to
    # navigate multiple structures in Mercury
    ciffile.header = filename
    writer.cf.data = {filename : ciffile}
    writer.write_file(filename)
    # Restore the Structure ignore_H_atoms setting
    Structure.ignore_H_atoms = ignore_H_setting

def get_multiple_CIFs_from_trajectory(Structure, result, decimals=3,
        filename_root="plot"):
    trajectory = result["trajectories"]
    best_particle = np.argmin(trajectory[-1][2])

    external = []
    internal = []
    chi_2 = []
    for t in trajectory:
        external.append(t[0][best_particle])
        internal.append(t[1][best_particle])
        chi_2.append(t[2][best_particle])

    external = np.vstack(external)
    internal = np.vstack(internal)
    chi_2 = np.asarray(chi_2)
    frac = zm_to_cart.get_asymmetric_coords_from_numpy(Structure, external,
                                                        internal)
    frac = np.around(frac, decimals)
    save_CIF_of_best_result(Structure, {"external" : external[0].reshape(1,-1),
                                    "internal" : internal[0].reshape(1,-1),
                                    "chi_2" : chi_2[0].reshape(1),
                                    "GALLOP Iter": 0},
                                    filename_root=filename_root)

    lines = []
    atom_labels = []
    atom_index = 1e6
    with open(glob.glob("plot*.cif")[0], "r") as init_cif:
        for i, line in enumerate(init_cif):
            splitline = list(filter(None,line.strip().split(" ")))
            if splitline[0] != "H":
                lines.append(line)
                if i >= atom_index:
                    atom_labels.append([splitline[0:3]]+[[splitline[-1]]])
            elif not Structure.ignore_H_atoms:
                lines.append(line)
                if i >= atom_index:
                    atom_labels.append([splitline[0:3]]+[[splitline[-1]]])
            if "_atom_site_occupancy" in line:
                atom_index = i+1
        init_cif.close()
    os.remove(glob.glob("plot*.cif")[0])
    header = lines[:atom_index]

    cifs = []
    for f in frac.astype(str).tolist():
        atoms = []
        for i, atom in enumerate(f):
            atoms.append(" ".join(atom_labels[i][0]
                        + atom + atom_labels[i][1])+"\n")

        cifs.append(" ".join(header) + " ".join(atoms))

    return cifs

def save_animation_from_trajectory(result, Structure, cifs=None,
    just_full_cell=False, just_asymmetric_unit=False, filename_root="animation",
    interval=30, return_view=False, height=420, width=680):
    if cifs is None:
        cifs = get_multiple_CIFs_from_trajectory(Structure, result)
    if just_full_cell:
        view = py3Dmol.view(height=height, width=width)
        view.addModelsAsFrames("\n".join(cifs), 'cif',
                        {"doAssembly" : True,
                        "normalizeAssembly":True,
                        'duplicateAssemblyAtoms':True})
        view.animate({'loop': 'forward', 'interval': interval})
        view.setStyle({'model':0},{'sphere':{"scale":0.15},
                                    'stick':{"radius":0.25}})
        view.addUnitCell()
        view.zoomTo()
        t = view.js()
        f = open(filename_root
                + f'_iter_{result["GALLOP Iter"]+1}_cell_anim.html','w')
        f.write(t.startjs)
        f.write(t.endjs)
        f.close()
    elif just_asymmetric_unit:
        view = py3Dmol.view(height=height, width=width)
        view.addModelsAsFrames("\n".join(cifs), 'cif',
                        {"doAssembly" : False,
                        "normalizeAssembly":True,
                        'duplicateAssemblyAtoms':True})
        view.animate({'loop': 'forward', 'interval': interval})
        view.setStyle({'model':0},{'sphere':{"scale":0.15},
                                    'stick':{"radius":0.25}})
        view.zoomTo()
        t = view.js()
        f = open(filename_root
                + f'_iter_{result["GALLOP Iter"]+1}_asym_anim.html', 'w')
        f.write(t.startjs)
        f.write(t.endjs)
        f.close()
    else:
        view = py3Dmol.view(linked=False, viewergrid=(1,2),height=height,
                            width=width)
        view.addModelsAsFrames("\n".join(cifs), 'cif',
                        {"doAssembly" : True,
                        "normalizeAssembly":True,
                        'duplicateAssemblyAtoms':True},
                        viewer=(0,0))
        view.addModelsAsFrames("\n".join(cifs), 'cif',
                        {"doAssembly" : False,
                        "normalizeAssembly":True,
                        'duplicateAssemblyAtoms':True},
                        viewer=(0,1))
        view.animate({'loop': 'forward', 'interval': interval})
        view.setStyle({'model':0},{'sphere':{"scale":0.15},
                                    'stick':{"radius":0.25}})
        view.addUnitCell(viewer=(0,0))
        view.zoomTo()
        t = view.js()
        f = open(filename_root
                + f'_iter_{result["GALLOP Iter"]+1}_both_anim.html', 'w')
        f.write(t.startjs)
        f.write(t.endjs)
        f.close()
    if return_view:
        return view.zoomTo()