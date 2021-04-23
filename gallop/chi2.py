import gallop.intensities as intensities
import gallop.zm_to_cart as zm_to_cart
import gallop.tensor_prep as tensor_prep

import torch

@torch.jit.script
def calc_chisqd(intensities, inverse_covariance_matrix, observed_intensities,
    sum_1_1):
    """
    Given sets of calculated intensities, the Pawley-derived inverse covariance
    matrix, and the extracted observed intensities, calculate chi2 according to
    the equation described by David et al.

    Args:
        intensities (Tensor): Tensor of shape (n_samples, n_peaks) containing
            the calculated intensity of each hkl
        inverse_covariance_matrix (Tensor): Tensor of shape (n_peaks, n_peaks)
            obtained from the Pawley fit of the diffraction data
        observed_intensities (Tensor): Tensor of shape (n_peaks,) - observed
            intensities obtained by Pawley fit
        sum_1_1 (Tensor): Pre-calculated first term of the scaling factor,
            equivalent to:
            torch.mv(inverse_covariance_matrix, observed_intensities)

    Returns:
        Tensor: the chi2 value for each calculated pattern
    """
    sum_1_2 = torch.einsum("ij,j->i",intensities,sum_1_1)

    sum_2_1 = torch.einsum("ij,kj->ki",inverse_covariance_matrix, intensities)
    sum_2_2 = torch.einsum("ij,ij->i",intensities,sum_2_1)

    # Scaling factor for calculated intensities.
    scale = (sum_1_2 / sum_2_2).reshape(-1,1)

    # Difference between observed and scaled calculated intensities
    diff = observed_intensities.repeat(intensities.shape[0],1)\
            - (intensities*scale)

    # Finally calculate chisqd - d.A.d
    chi_1 = torch.einsum("ij,kj->ki",inverse_covariance_matrix,diff) # A.d
    chi_2 = torch.einsum("ij,ij->i",diff,chi_1) # d.(A.d)

    return chi_2 / (intensities.shape[1] - 2)



def get_chi_2(external, internal, position, rotation, torsion, initial_D2,
        zmatrices_degrees_of_freedom, bond_connection, angle_connection,
        torsion_connection, torsion_refineable_indices, lattice_inv_matrix,
        init_cart_coords, hkl, intensity_calc_prefix_fs,
        intensity_calc_prefix_fs_asymmetric, nsamples_ones, affine_matrices,
        centrosymmetric, space_group_number, observed_intensities,
        chisqd_scale_sum_1_1, inverse_covariance_matrix):
    """
    Convenience function to get chi2 directly from external/internal rather
    than performing the three steps required:
    1. coords, 2. intensities, 3. chi2

    The required arguments are produced and packaged into a dictionary by the
    gallop.tensor_prep code

    Args:
        external (Tensor): The external degrees of freedom
        internal (Tensor): The internal degrees of freedom
        position (list): List of tensors with the indices of the positional
            degrees of freedom (dtype=torch.long)
        rotation (list): List of tensors with the indices of the rotational
            degrees of freedom (dtype=torch.long)
        torsion (list): List of tensors with the indices of the torsional
            degrees of freedom (dtype=torch.long)
        initial_D2 (list): List of tensors of the initial_D2 matrices for each
            ZM with >0 internal DOFs
        zmatrices_degrees_of_freedom (list): List of ints of the total number
            of degrees of freedom for each Z-matrix
        bond_connection (list): List of tensors with the indices of the bonded
            atoms obtained from the ZMs. Only for ZMs with > 0 torsions.
            (dtype=torch.long)
        angle_connection (list): List of tensors with the indices of the angle
            atoms obtained from the ZMs. Only for ZMs with > 0 torsions.
            (dtype=torch.long)
        torsion_connection (list): List of tensors with the indices of the
            torsion atoms obtained from the ZMs. Only for ZMs with > 0 torsions.
            (dtype=torch.long)
        torsion_refineable_indices (list): List of tensors with the indices of
            the refineable torsion angles obtained from the ZMs. Only for ZMs
            with > 0 torsions. (dtype=torch.long)
        lattice_inv_matrix (Tensor): Inverse of the lattice matrix
            representation to convert Cartesian -> fractional coords
        init_cart_coords (list): List of tensors with Cartesian coordinates
            of rigid bodies (dtype=torch.long)
        hkl (Tensor): The Miller indices of the reflections
        intensity_calc_prefix_fs (Tensor): The atomic scattering factors, occus
            and Debye-Waller factors (optional) for all atoms in the unit cell
        intensity_calc_prefix_fs_asymmetric (Tensor): The atomic scattering
            factors, occus and Debye-Waller factors (optional) for atoms in the
            asymmetric unit.
        nsamples_ones (Tensor): A tensor of shape (n_samples, n_atoms, 1) that
            is required for the fall-back method of intensity calculation
        affine_matrices (Tensor): A set of N 4x4 matrices obtained from the
            space group used in the fall-back method of intensity calculation,
            where N is the number of symmetry equivalent positions in the unit
            cell.
        centrosymmetric (bool): True if the space group is centrosymmetric
        space_group_number (int): The number of the space group
        observed_intensities (Tensor): The observed intensities obtained from
            a Pawley fit
        chisqd_scale_sum_1_1 (Tensor): Pre-calculated term used in a scaling
            factor in chi_2 calculation
        inverse_covariance_matrix (Tensor): The inverse_covariance_matrix
            obtained from the Pawley fit

    Returns:
        Tensor: The chi_2 values corresponding to the n_samples
    """
    asymmetric_frac_coords = zm_to_cart.get_asymmetric_coords(
                                external, internal, position, rotation, torsion,
                                initial_D2, zmatrices_degrees_of_freedom,
                                bond_connection, angle_connection,
                                torsion_connection, torsion_refineable_indices,
                                lattice_inv_matrix, init_cart_coords)

    calculated_intensities = intensities.calculate_intensities(
                                asymmetric_frac_coords, hkl,
                                intensity_calc_prefix_fs,
                                intensity_calc_prefix_fs_asymmetric,
                                nsamples_ones, affine_matrices, centrosymmetric,
                                space_group_number)

    chisqd = calc_chisqd(calculated_intensities, inverse_covariance_matrix,
                        observed_intensities, chisqd_scale_sum_1_1)

    return chisqd


def get_chi2_from_CIF(Structure, n_reflections=None, include_dw_factors=True,
    dtype=torch.float32, device=None):
    """
    Read in a CIF and calculate the chi_2 value given the data in the Structure
    object.

    Args:
        Structure (class): GALLOP Structure object with the hkls, observed
            intensities and inverse covariance matrix present.
        n_reflections (int, optional): If None, uses all available reflections
            for the chi_2 calculation. Defaults to None.
        include_dw_factors (bool, optional): Include Debye-Waller factors in
            the intensity calculation. Defaults to True.
        dtype (torch dtype, optional): The data type to use.
            Defaults to torch.float32.
        device (torch device, optional): Where to run the calculations.
            If None, checks to see if a GPU is present and runs on it if so.
            Defaults to None.

    Returns:
        numpy array: The chi_2 value obtained from the CIF
    """
    tensors = tensor_prep.get_all_required_tensors(
                Structure, device=device, dtype=dtype,
                n_reflections=n_reflections, verbose=False,
                include_dw_factors=include_dw_factors, from_CIF=True
                )
    centrosymmetric = tensors["centrosymmetric"]
    space_group_number = tensors["space_group_number"]
    asymmetric_frac_coords = tensors["asymmetric_frac_coords"]
    hkl = tensors["hkl"]
    i_calc_prefix_fs = tensors["intensity_calc_prefix_fs"]
    i_calc_prefix_fs_asymmetric = tensors["intensity_calc_prefix_fs_asymmetric"]
    nsamples_ones = tensors["nsamples_ones"]
    affine_matrices = tensors["affine_matrices"]
    inverse_covariance_matrix = tensors["inverse_covariance_matrix"]
    observed_intensities = tensors["observed_intensities"]
    chisqd_scale_sum_1_1 = tensors["chisqd_scale_sum_1_1"]


    calculated_intensities = intensities.calculate_intensities(
                        asymmetric_frac_coords, hkl, i_calc_prefix_fs,
                        i_calc_prefix_fs_asymmetric, nsamples_ones,
                        affine_matrices, centrosymmetric, space_group_number)

    chisqd = calc_chisqd(calculated_intensities, inverse_covariance_matrix,
                        observed_intensities, chisqd_scale_sum_1_1)

    del tensors

    return chisqd.detach().cpu().numpy()
