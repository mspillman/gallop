# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Fast conversion from internal to Cartesian coordinates.
"""

import torch
import gallop.tensor_prep as tensor_prep


@torch.jit.script
def modify_D2(tors, D2, refineable_torsion_indices):
    """
    Modify a base "D2" matrix for use in the NERF approach for
    internal -> Cartesian. See function zm_to_cart for more details.

    Args:
        tors (Tensor): The values of the refineable torsion angles
        D2 (Tensor): The base D2 matrix
        refineable_torsion_indices (Tensor): The indices of the refineable
            torsion angles

    Returns:
        Tensor: A modified D2 matrix including the updated torsion angles
    """
    cos_phi = torch.cos(tors)
    sin_phi = torch.sin(tors)
    new_D2 = D2.clone()
    new_D2[:,:,1][:,refineable_torsion_indices] = \
        torch.mul(D2[:,:,1][:,refineable_torsion_indices], cos_phi)
    new_D2[:,:,2][:,refineable_torsion_indices] = \
        torch.mul(D2[:,:,2][:,refineable_torsion_indices], sin_phi)
    return new_D2

@torch.jit.script
def zm_to_cart(torsions, init_D2, bond_connect, angle_connect, torsion_connect,
    refineable_torsion_indices):
    """
    This function uses the Natural Extension Reference Frame method to convert
    from internal (Z-matrix) to Cartesian coordinates.
    See https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.20237 for details.
    The names of the variables used are roughly in line with paper.

    It currently accepts only torsion angles as refineable parameters.

    Args:
        torsions (Tensor): Tensor of refineable torsion angle values
            (generally, this is what is being optimised)
        init_D2 (Tensor): Generated from the original z-matrix. This is
            modified by the refineable torsion angles. D2 is a term used in the
            NeRF ZM->Cart paper.
        bond_connect (Tensor): Tensor containing the indices of directly bonded
            atoms
        angle_connect (Tensor): Tensor containing the indices of atoms involved
            in angles
        torsion_connect (Tensor): Tensor containing the indices of atoms
            involved in torsions
        refineable_torsion_indices (Tensor): Tensor containing the indices of
            refineable torsion angles.

    Returns:
        cart (Tensor): Cartesian coordinates of the atoms given by a Z-matrix
                        description. Shape (n_samples, n_atoms, 3)
    """
    D2 = modify_D2(torsions, init_D2, refineable_torsion_indices)
    cart_list = [torch.clone(D2[:,0,:]),
                torch.clone(D2[:,1,:]),
                torch.clone(D2[:,2,:])]
    for i in range(D2.shape[1]-3):
        i+=3 # torch.jit doesn't work with range(3, atoms)
        C = cart_list[bond_connect[i]]
        B = cart_list[angle_connect[i]]
        A = cart_list[torsion_connect[i]]
        c_minus_b = C - B
        bc = c_minus_b.div(c_minus_b.norm(p=2, dim=-1, keepdim=True))
        AB = B - A
        n_1 = torch.cross(AB, bc)
        n = n_1.div(n_1.norm(p=2, dim=-1, keepdim=True))
        n_bc = torch.cross(n, bc)
        M = torch.stack((bc, n_bc, n), dim=-1)
        cart_list.append(torch.einsum("bij,bj->bi",M, D2[:,i]) + C)
    cart = torch.stack(cart_list, dim=1)
    return cart

@torch.jit.script
def get_rotation_matrices(rot):
    """
    Get rotation matrices from quaternions
    Obtained formulae from here:
    https://uk.mathworks.com/help/robotics/ref/quaternion.rotmat.html

    Args:
        rot (Tensor): unit quaternions representing the rotation of a Single
            z-matrix

    Returns:
        Tensor: Tensor with shape (n_samples, 3, 3) containing the rotation
            matrices equivalent to the quaternions
    """
    a = rot[:,0].reshape(rot.shape[0],1)
    b = rot[:,1].reshape(rot.shape[0],1)
    c = rot[:,2].reshape(rot.shape[0],1)
    d = rot[:,3].reshape(rot.shape[0],1)

    row_1_1 = 1 - 2*(c**2 + d**2)
    row_1_2 = 2*(b*c - d*a)
    row_1_3 = 2*(b*d + c*a)

    row_2_1 = 2*(b*c + d*a)
    row_2_2 = 1 - 2*(b**2 + d**2)
    row_2_3 = 2*(c*d - b*a)

    row_3_1 = 2*(b*d - c*a)
    row_3_2 = 2*(c*d + b*a)
    row_3_3 = 1 - 2*(b**2 + c**2)

    #R = torch.cat([row_1_1,row_1_2,row_1_3,
    #                row_2_1,row_2_2,row_2_3,
    #                row_3_1,row_3_2,row_3_3],dim=1).reshape(rot.shape[0], 3, 3)
    #R = torch.transpose(R, 2, 1)
    row_1 = torch.cat((row_1_1,row_1_2,row_1_3), dim=1)
    row_2 = torch.cat((row_2_1,row_2_2,row_2_3), dim=1)
    row_3 = torch.cat((row_3_1,row_3_2,row_3_3), dim=1)
    R = torch.cat((row_1.unsqueeze(2), row_2.unsqueeze(2), row_3.unsqueeze(2)),
                    dim=2)
    return R

@torch.jit.script
def rotate_and_translate(cart_coords, R, translation, lattice_inv_matrix):
    """
    Given a set of Cartesian coordinates, rotate them, convert them to
    fractional coordinates then translate them.

    Args:
        cart_coords (Tensor): Cartesian coordinates of the atoms, usually
            obtained by a ZM->Cart conversion.
        R (Tensor): Rotation matrices (generally obtained from quaternions
            which are the refineable parameters)
        translation (Tensor): Describes the translation of molecues/fragments
        lattice_inv_matrix (Tensor): Inverse lattice matrix to convert from
            Cartesian to fractional coords

    Returns:
        Tensor: Tensor of shape (nsamples, natoms, 3) containing the
                fractional coordinates of the atoms after rotation and
                translation.
    """
    # Rotate the Cartesian coordinates using rotation matrices, R
    rot_cart = torch.einsum("bjk,bij->bik", R, cart_coords)
    # Convert to fractional coords
    rot_cart = torch.einsum("jk,bij->bik",lattice_inv_matrix,rot_cart)
    # Translate the sites. Must use repeat_interleave rather than repeat here.
    # See Torch documentation here:
    # https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html
    # >>> x = torch.tensor([1, 2, 3])
    # >>> x.repeat_interleave(2)
    # tensor([1, 1, 2, 2, 3, 3])
    # >>> x.repeat(2)
    # tensor([1, 2, 3, 1, 2, 3])
    #translations = torch.repeat_interleave(translation,
    #        repeats=cart_coords.shape[1], dim=0).reshape(translation.shape[0],
    #                                                cart_coords.shape[1], 3)
    fractional_coords = rot_cart + translation.view(translation.shape[0], 1, 3)
    return fractional_coords


def get_asymmetric_coords(external, internal, position, rotation, torsion,
    initial_D2, zmatrices_degrees_of_freedom, bond_connection, angle_connection,
    torsion_connection, torsion_refineable_indices, lattice_inv_matrix,
    init_cart_coords):
    """
    Take in a set of external and internal degrees of freedom and generate the
    fractional coordinates of
    all atoms in the asymmetric unit. Uses the NERF method for converting from
    internal -> Cartesian coordinates.

    Args:
        external (Tensor): Tensor of shape (n_samples, n_ext_DOF) - The external
            degrees of freedom for the current problem. Where multiple
            z-matrices are needed, then the order is
            pos_1, pos_2 ... pos_n, rot_1, rot_2, ..., rot_n
        internal (Tensor): Tensor of shape (n_samples, n_int_DOF) - The internal
            degrees of freedom for the current problem. Where multiple
            z-matrices are needed, then the order is tor_1, tor_2, ..., tor_n
        position (List): List of lists, containing the indices of external
            corresponding to position for each zmatrix
        rotation (List): List of lists, containing the indices of external
            corresponding to rotation for each zmatrix
        torsion (List): List of lists, containing the indices of external
            corresponding to torsions for each zmatrix
        zmatrices_degrees_of_freedom (List): List of integers, describing the
            number of degrees of freedom for each zmatrix.
            Note that rotations are considered to require 4 DoF due to the use
            of quaternions. This is in contrast to normally reported values,
            where Euler angles are usually considered instead.
        initial_D2 (List): List of tensors of shape (n_samples, n_atoms) for
            each flexible zmatrix. Needed for internal -> Cartesian
        bond_connection (List): List of tensors of shape (n_samples, n_atoms)
            for each flexible zmatrix. Needed for internal -> Cartesian
        angle_connection (List): List of tensors of shape (n_samples, n_atoms)
            for each flexible zmatrix. Needed for internal -> Cartesian
        torsion_connection (List): List of tensors of shape (n_samples, n_atoms)
            for each flexible zmatrix. Needed for internal -> Cartesian
        torsion_refineable_indices (List): List of tensors that contain the
            indices of the refineable torsions for each zmatrix
        lattice_inv_matrix (Tensor): Inverse of the matrix representation of the
            lattice. Needed to for Cartesian -> fractional
        init_cart_coords (List): List of Tensors, with the Cartesian coordinates
            of the Z-matrices. Needed for rigid-body ZMs.

    Returns:
        Tensor: Tensor of shape (n_samples, n_atoms_asymmetric), which is the
                fractional coordinates of all atoms in the asymmetric_unit
    """
    asymmetric_unit = []
    for i, zm_dof in enumerate(zmatrices_degrees_of_freedom):
        translation = external[:,position[i]].view(external.shape[0],
                                            1, len(position[i])) # Translations
        if zm_dof >= 7:
            rot_in = external[:,rotation[i]] # Quaternions for rotation
            # Ensure quaternions are unit quaterions then convert to matrices
            rot = rot_in.div(rot_in.norm(p=2,dim=-1,keepdim=True))
            R = get_rotation_matrices(rot)

            if zm_dof > 7:
                # Convert flexible ZM to Cartesian coords
                cart_coords = zm_to_cart(internal[:,torsion[i]], initial_D2[i],
                                        bond_connection[i], angle_connection[i],
                                        torsion_connection[i],
                                        torsion_refineable_indices[i])
            else:
                # Rigid body
                cart_coords = init_cart_coords[i]

            fractional_coords = rotate_and_translate(cart_coords, R,
                                                translation, lattice_inv_matrix)
            asymmetric_unit.append(fractional_coords)
        else:
            # Single atoms/ions
            fractional_coords = translation
            asymmetric_unit.append(fractional_coords)

    # Collect all atoms in asymmetric unit
    asymmetric_frac_coords = torch.cat(asymmetric_unit, dim=1)
    return asymmetric_frac_coords


def get_asymmetric_coords_from_numpy(Structure, external, internal,
    n_samples=None, device=None, dtype=torch.float32, verbose=True):
    """
    Convenience function used mostly for saving a CIF at the end of a run.

    Args:
        Structure (class): Structure object
        external (numpy array): external degrees of freedom
        internal (numpy array): internal degrees of freedom
        n_samples (int, optional): The number of samples. Defaults to None,
            which results in a single set of coordinates
        device (torch device, optional): If None, will check to see if a GPU
            is available and run on that if so. Defaults to None.
        dtype (torch dtype, optional): The datatype to use. Defaults to
            torch.float32.
        verbose (bool, optional): Print out information. Defaults to True.

    Returns:
        numpy array: fractional coordinates of the asymmetric unit
    """
    if n_samples == None:
        n_samples = 1

    # Load the tensors needed
    tensors = tensor_prep.get_all_required_tensors(
                    Structure, external=external, internal=internal,
                    n_samples=n_samples, device=device, dtype=dtype,
                    verbose=verbose)

    asymmetric_frac_coords = get_asymmetric_coords(**tensors["zm"])


    return asymmetric_frac_coords.detach().cpu().numpy()
