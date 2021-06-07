# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for working with files.
"""
import pickle
import numpy as np
import pymatgen as pmg


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