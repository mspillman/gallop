# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions to generate all of the tensors needed for local optimisation
"""

import torch
import numpy as np


def get_zm_related_tensors(Structure, n_samples, dtype, device):
    """
    Get tensors about the ZMs from the numpy arrays in the Structure.

    The external and internal DoF tensors may contain information
    pertaining to multiple fragments. The information needs to be indexed
    such that positions, rotations and torsions are assigned consistently
    and correctly. Example below has 2 fragments, with 6 torsions each.
    Each fragment therefore requires:
        3 x position, 4 x quaternions, 6 x torsions
    The layout of the DoF of the tensors is as follows:
        external = n_samples*[pos_1_x, pos_1_y, pos_1_z, pos_2_x, pos_2_y,
                                pos_2_z, rot_1_q1, rot_1_q2, rot_1_q3, rot_1_q4,
                                rot_2_q1, rot_2_q2, rot_2_q3, rot_2_q4]
        internal = n_samples*[tor_1_1, tor_1_2, tor_1_3, tor_1_4, tor_1_5,
                                tor_1_6, tor_2_1, tor_2_2, tor_2_3, tor_2_4,
                                tor_2_5, tor_2_6]
    This function makes lists of numpy arrays: position_indices,
    rotation_indices, torsion_indices that denote what information
    pertains to each fragment. In the above example:
        position_indices = [[0,1,2], [3,4,5]]
        rotation_indices = [[6,7,8,9], [10,11,12,13]]
        torsion_indices  = [[0,1,2,3,4,5], [6,7,8,9,10,11]]

    Args:
        Structure (class): GALLOP structure object
        n_samples (int): Number of simultaneous samples to load
        dtype (torch dtype): dtype to use for tensors
        device (torch device): device to move tensors to

    Returns:
        dict: Dictionary of the tensors. The complete dict of values returned
        by this function is given below:
            tensors = {"zmatrices_degrees_of_freedom" : zm_degrees_of_freedom,
                    "position" : zm_position_indices,
                    "rotation" : zm_rotation_indices,
                    "torsion" : zm_torsion_indices,
                    "initial_D2" : init_D2,
                    "torsion_refineable_indices" : torsion_refineable_indices,
                    "bond_connection" : bond_connection,
                    "angle_connection" : angle_connection,
                    "torsion_connection" : torsion_connection,
                    "init_cart_coords" : init_cart_coords,
                    "nsamples_ones" : nsamples_ones
                    }
    """
    position_indices = []
    rotation_indices = []
    torsion_indices = []
    zm_degrees_of_freedom = []
    init_D2 = []
    torsion_refineable_indices = []
    bond_connection = []
    angle_connection = []
    torsion_connection = []
    init_cart_coords = []
    n_atoms_asymmetric = 0
    i = 0
    # First loop over each of the z-matrices and get information about
    # connectivity, degrees of freedom etc
    for i, zm in enumerate(Structure.zmatrices):
        # First generate the indices needed
        if i == 0:
            position_indices.append(np.arange(zm.position_degrees_of_freedom))
            rotation_indices.append(np.arange(zm.rotation_degrees_of_freedom))
            torsion_indices.append(np.arange(zm.internal_degrees_of_freedom))
        else:
            try:
                max_pos = np.hstack(position_indices).max()
            except ValueError:
                max_pos = -1
            try:
                max_rot = np.hstack(rotation_indices).max()
            except ValueError:
                max_rot = -1
            try:
                max_tor = np.hstack(torsion_indices).max()
            except ValueError:
                max_tor = -1
            position_indices.append(np.arange(zm.position_degrees_of_freedom)
                                            + max_pos + 1)
            rotation_indices.append(np.arange(zm.rotation_degrees_of_freedom)
                                            + max_rot + 1)
            torsion_indices.append(np.arange(zm.internal_degrees_of_freedom)
                                            + max_tor + 1)
        zm_degrees_of_freedom.append(zm.degrees_of_freedom)

        # Now get initial D2 matrix and torsion refineable indices for
        # internal -> Cartesian conversion
        # and get initial Cartesian coordinates for rigid bodies. See (S)NeRF
        # paper and gallop.z_matrix for more details.
        if Structure.ignore_H_atoms:
            init_D2_stacked = zm.initial_D2_no_H.reshape(1,
                        zm.initial_D2_no_H.shape[0],
                        zm.initial_D2_no_H.shape[1]).repeat(n_samples,axis=0)
            torsion_refineable_indices.append(torch.from_numpy(
                zm.torsion_refineable_indices_no_H).type(torch.long).to(device))
            if zm.degrees_of_freedom == 7:
                init_cart_coords.append(torch.from_numpy(
                        zm.initial_cartesian_no_H.reshape(1,
                        zm.initial_cartesian_no_H.shape[0],
                        zm.initial_cartesian_no_H.shape[1])
                        .repeat(n_samples,axis=0)).type(dtype).to(device))
            else:
                init_cart_coords.append([])
        else:
            init_D2_stacked = zm.initial_D2.reshape(1,
                                zm.initial_D2.shape[0],
                                zm.initial_D2.shape[1]).repeat(n_samples,axis=0)
            torsion_refineable_indices.append(
                torch.from_numpy(
                    zm.torsion_refineable_indices).type(torch.long).to(device)
                )
            if zm.degrees_of_freedom == 7:
                init_cart_coords.append(torch.from_numpy(
                        zm.initial_cartesian.reshape(1,
                            zm.initial_cartesian.shape[0],
                            zm.initial_cartesian.shape[1]).repeat(
                                n_samples,axis=0)).type(dtype).to(device)
                                )
            else:
                init_cart_coords.append([])

        init_D2.append(torch.from_numpy(init_D2_stacked).type(dtype).to(device))

        # Next get the molecular connectivity information
        if Structure.ignore_H_atoms:
            bond_connection.append(
                torch.from_numpy(
                    zm.bond_connection_no_H).type(torch.long).to(device)
                )
            angle_connection.append(
                torch.from_numpy(
                    zm.angle_connection_no_H).type(torch.long).to(device)
                )
            torsion_connection.append(
                torch.from_numpy(
                    zm.torsion_connection_no_H).type(torch.long).to(device)
                )
        else:
            bond_connection.append(
                torch.from_numpy(zm.bond_connection).type(torch.long).to(device)
                )
            angle_connection.append(
                torch.from_numpy(
                    zm.angle_connection).type(torch.long).to(device)
                )
            torsion_connection.append(
                torch.from_numpy(
                    zm.torsion_connection).type(torch.long).to(device)
                )
        if Structure.ignore_H_atoms:
            n_atoms_asymmetric += zm.coords_no_H.shape[0]
        else:
            n_atoms_asymmetric += zm.coords.shape[0]

    # Now correct the external DoF indices for rotations
    # The external input has the following layout:
    # pos_zm_1, pos_zm_2, ... , pos_zm_n, rot_zm_1, rot_zm_2, ... , rot_zm_n
    # so the previously generated indices need to be corrected for the total
    # number of positional indexes that are also included.
    for r in rotation_indices:
        try:
            r+=np.hstack(position_indices).max()+1
        except ValueError:
            pass

    # Add this information to the Structure object for easy access
    Structure.degrees_of_freedom = zm_degrees_of_freedom
    Structure.position_indices   = position_indices
    Structure.rotation_indices   = rotation_indices
    Structure.torsion_indices    = torsion_indices

    # Create the lists of PyTorch tensors needed. Tensors used for indexing must
    # be of type torch.long
    zm_position_indices = []
    zm_rotation_indices = []
    zm_torsion_indices  = []
    for p in position_indices:
        zm_position_indices.append(
                                torch.from_numpy(p).type(torch.long).to(device))
    for r in rotation_indices:
        zm_rotation_indices.append(
                                torch.from_numpy(r).type(torch.long).to(device))
    for t in torsion_indices:
        zm_torsion_indices.append(
                                torch.from_numpy(t).type(torch.long).to(device))

    # nsamples_ones needed for fall-back Structure factor calculation method.
    # It is concatenated with the x,y,z fractional coordinates, which then
    # allows for multiplication with the affine matrices to generate symmetry
    # equivalent positions.
    nsamples_ones = torch.ones(n_samples,
                                n_atoms_asymmetric,
                                1).type(dtype).to(device)

    tensors = {"zmatrices_degrees_of_freedom" : zm_degrees_of_freedom,
                "position" : zm_position_indices,
                "rotation" : zm_rotation_indices,
                "torsion" : zm_torsion_indices,
                "initial_D2" : init_D2,
                "torsion_refineable_indices" : torsion_refineable_indices,
                "bond_connection" : bond_connection,
                "angle_connection" : angle_connection,
                "torsion_connection" : torsion_connection,
                "init_cart_coords" : init_cart_coords,
                "nsamples_ones" : nsamples_ones}

    return tensors

def get_dw_factors(Structure):
    """
    Get the Debye-Waller factors for inclusion in the intensity calculations

    Args:
        Structure (class): GALLOP structure object

    Returns:
        dict: Debye-Waller factors for each atom in a ZM.
    """
    dw_factors = {}
    for zm in Structure.zmatrices:
        dw_factors.update(zm.dw_factors)
    return dw_factors

def get_data_related_tensors(Structure, n_reflections, dtype, device,
    verbose=True):
    """
    Get tensors about the PXRD data from the numpy arrays in the Structure

    Args:
        Structure (class): GALLOP structure object
        n_reflections ([type]): Number of reflections to consider for chi_2
            calculation
        dtype (torch dtype): dtype to use for tensors
        device (torch device): device to move tensors to
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        dict: Dictionary of the tensors
    """
    if n_reflections is not None:
        assert n_reflections > 0, "Cannot optimise with <= 0 reflections"
        if n_reflections >  len(Structure.hkl):
            n_reflections = len(Structure.hkl)
    else:
        n_reflections = len(Structure.hkl)
    if verbose:
        total_ref = len(Structure.hkl)
        res = Structure.get_resolution(Structure.twotheta[n_reflections-1], )
        print("Using",n_reflections,"of",total_ref,"available reflections.")
        print("Resolution with",n_reflections,"reflections:",res)

    tensors = {}
    tensors["affine_matrices"] = torch.from_numpy(
                        Structure.affine_matrices).type(dtype).to(device)
    tensors["hkl"] = torch.from_numpy(
                        Structure.hkl[:n_reflections].T).type(dtype).to(device)
    tensors["lattice_inv_matrix"] = torch.from_numpy(
                np.copy(Structure.lattice.inv_matrix)).type(dtype).to(device)


    inverse_covariance_matrix = Structure.inverse_covariance_matrix
    inverse_covariance_matrix = inverse_covariance_matrix[:n_reflections,
                                                            :n_reflections]
    tensors["inverse_covariance_matrix"] = torch.from_numpy(
                            inverse_covariance_matrix).type(dtype).to(device)
    tensors["observed_intensities"] = torch.from_numpy(
                Structure.intensities[:n_reflections]).type(dtype).to(device)
    tensors["chisqd_scale_sum_1_1"] = torch.mv(
        tensors["inverse_covariance_matrix"], tensors["observed_intensities"])
    return tensors

def get_all_required_tensors(Structure, external=None, internal=None,
    n_samples=10000, device=None, dtype=None, n_reflections=None, verbose=True,
    include_dw_factors=True, requires_grad=True, from_CIF=False):
    """
    Gather all of the information needed to run calculations on the GPU. This
    will differ depending on if the input is a set of external/internal degrees
    of freedom (standard) or if the input is directly from a CIF (used for
    mostly for debugging and comparison to DASH-derived values)

    Args:
        Structure (Structure Class): Class containing all of the data and
            z-matrices
        external (Numpy array, optional): Either a numpy array of the external
            DoF or None.
            If None, then the external DoF will be generated randomly.
            Defaults to None.
        internal (Numpy array, optional): Either a numpy array of the internal
            DoF or None.
            If None, then the external DoF will be generated randomly.
            Defaults to None.
        n_samples (int, optional): If external is None, sets how many samples
            of external DoF to take. Defaults to 10000.
        device (torch device, optional): Where to run the code. If None, will
            check for a GPU and run on that if one is available.
        n_reflections (int, optional): Number of reflections to consider in
            chi^2 calculations. If None, will use all available.
        verbose (bool, optional): Print out some information or not. Defaults
            to True.
        include_dw_factors (bool, optional): Include Debye-Waller factors in
            the intensity calculations. Defaults to True.
        requires_grad (bool, optional): Gradients are required if parameters
            (external, internal) need to be optimised. Defaults to True.
        from_CIF (bool, optional): If reading coordinates from a CIF, need to
            change which tensors are required. Defaults to False.

    Returns:
        dict: A dictionary of the required Tensors
    """
    if len(Structure.zmatrices) == 0:
        print("Error - no Z-matrices have been added to the Structure")
        return None
    else:
        if not hasattr(Structure, "total_internal_degrees_of_freedom"):
            Structure.get_total_degrees_of_freedom()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.float32

    if external is None:
        pos = torch.rand(
                    (n_samples, Structure.total_position_degrees_of_freedom))
        rot = torch.rand(
                    (n_samples, Structure.total_rotation_degrees_of_freedom))
        rot -= 0.5
        rot *= 2
        external = torch.cat([pos, rot], dim=1)
        external = external.type(dtype).to(device)
        if requires_grad:
            external.requires_grad = True
    else:
        n_samples = external.shape[0]
        try:
            external = torch.from_numpy(external).type(dtype).to(device)
            if requires_grad:
                external.requires_grad = True
        except TypeError:
            print("Can't convert input",external,"to pytorch tensor")

    if internal is None:
        tor = torch.rand((n_samples,
                        Structure.total_internal_degrees_of_freedom))
        tor -= 0.5
        tor *= 2*np.pi
        internal = tor.type(dtype).to(device)
        if requires_grad:
            internal.requires_grad = True
    else:
        error = "external.shape[0] != internal.shape[0]"
        assert n_samples == internal.shape[0], error
        try:
            internal = torch.from_numpy(internal).type(dtype).to(device)
            if requires_grad:
                internal.requires_grad = True
        except TypeError:
            print("Can't convert input",internal,"to pytorch tensor")

    if include_dw_factors:
        if from_CIF:
            dw_factors = Structure.CIF_dw_factors
        else:
            dw_factors = get_dw_factors(Structure)
    else:
        dw_factors = {}

    tensors =  {"external" : external,
                "internal" : internal,
                **get_data_related_tensors(Structure, n_reflections,
                                    dtype, device, verbose=verbose)
                }
    if from_CIF:
        if Structure.ignore_H_atoms:
            n_atoms_asymmetric = Structure.cif_frac_coords_no_H.shape[0]
            asymmetric_frac_coords = Structure.cif_frac_coords_no_H.reshape(1,
                                                        n_atoms_asymmetric,3)
        else:
            n_atoms_asymmetric = Structure.cif_frac_coords.shape[0]
            asymmetric_frac_coords = Structure.cif_frac_coords.reshape(1,
                                                        n_atoms_asymmetric,3)
        asymmetric_frac_coords = torch.from_numpy(asymmetric_frac_coords)
        asymmetric_frac_coords = asymmetric_frac_coords.type(dtype).to(device)
        nsamples_ones = torch.ones(n_samples,
                                n_atoms_asymmetric,
                                1).type(dtype).to(device)
        tensors = {
            **tensors,
            "asymmetric_frac_coords" : asymmetric_frac_coords,
            "nsamples_ones" : nsamples_ones,
        }
    else:
        tensors = {**tensors, **get_zm_related_tensors(Structure,
                                                    n_samples, dtype, device)}
    # Calculate terms that are used in the intensity calculations, which include
    # the atomic scattering factors etc.
    full_prefix = Structure.generate_intensity_calculation_prefix(
                        debye_waller_factors=dw_factors, just_asymmetric=False,
                        from_cif=from_CIF)[:n_reflections]
    asymm_prefix = Structure.generate_intensity_calculation_prefix(
                        debye_waller_factors=dw_factors, just_asymmetric=True,
                        from_cif=from_CIF)[:n_reflections]

    tensors["intensity_calc_prefix_fs"] = torch.from_numpy(
                                            full_prefix).type(dtype).to(device)
    tensors["intensity_calc_prefix_fs_asymmetric"] = torch.from_numpy(
                                            asymm_prefix).type(dtype).to(device)
    tensors["centrosymmetric"] = Structure.centrosymmetric
    tensors["space_group_number"] = Structure.sg_number
    return tensors
