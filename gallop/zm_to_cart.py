# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Fast conversion from internal to Cartesian coordinates.
"""

import torch
import gallop.tensor_prep as tensor_prep


@torch.jit.script
def modify_D2(tors, D2, refinable_torsion_indices):
    """
    Modify a base "D2" matrix for use in the NERF approach for
    internal -> Cartesian. See function zm_to_cart for more details.

    Args:
        tors (Tensor): The values of the refinable torsion angles
        D2 (Tensor): The base D2 matrix
        refinable_torsion_indices (Tensor): The indices of the refinable
            torsion angles

    Returns:
        Tensor: A modified D2 matrix including the updated torsion angles
    """
    cos_phi = torch.cos(tors)
    sin_phi = torch.sin(tors)
    new_D2 = D2.clone()
    new_D2[:,:,1][:,refinable_torsion_indices] = \
        torch.mul(D2[:,:,1][:,refinable_torsion_indices], cos_phi)
    new_D2[:,:,2][:,refinable_torsion_indices] = \
        torch.mul(D2[:,:,2][:,refinable_torsion_indices], sin_phi)
    return new_D2

@torch.jit.script
def zm_to_cart(torsions, init_D2, bond_connect, angle_connect, torsion_connect,
    refinable_torsion_indices):
    """
    This function uses the Natural Extension Reference Frame method to convert
    from internal (Z-matrix) to Cartesian coordinates.
    See https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.20237 for details.
    The names of the variables used are roughly in line with paper.

    It currently accepts only torsion angles as refinable parameters.

    Args:
        torsions (Tensor): Tensor of refinable torsion angle values
            (generally, this is what is being optimised)
        init_D2 (Tensor): Generated from the original z-matrix. This is
            modified by the refinable torsion angles. D2 is a term used in the
            NeRF ZM->Cart paper.
        bond_connect (Tensor): Tensor containing the indices of directly bonded
            atoms
        angle_connect (Tensor): Tensor containing the indices of atoms involved
            in angles
        torsion_connect (Tensor): Tensor containing the indices of atoms
            involved in torsions
        refinable_torsion_indices (Tensor): Tensor containing the indices of
            refinable torsion angles.

    Returns:
        cart (Tensor): Cartesian coordinates of the atoms given by a Z-matrix
                        description. Shape (n_samples, n_atoms, 3)
    """
    D2 = modify_D2(torsions, init_D2, refinable_torsion_indices)
    cart_list = torch.zeros_like(D2)
    cart_list[:,:3,:] = D2[:,:3,:]
    for i in range(D2.shape[1]-3):
        i+=3 # torch.jit doesn't work with range(3, atoms)
        C = cart_list[:,bond_connect[i]]
        B = cart_list[:,angle_connect[i]]
        A = cart_list[:,torsion_connect[i]]
        c_minus_b = C - B
        bc = c_minus_b.div(c_minus_b.norm(p=2, dim=-1, keepdim=True))
        AB = B - A
        n_1 = torch.cross(AB, bc)
        n = n_1.div(n_1.norm(p=2, dim=-1, keepdim=True))
        n_bc = torch.cross(n, bc)
        M = torch.stack((bc, n_bc, n), dim=-1)
        cart_list[:,i] = torch.einsum("bij,bj->bi",M, D2[:,i]) + C
    return cart_list

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
    a = rot[:,0].unsqueeze(1)
    b = rot[:,1].unsqueeze(1)
    c = rot[:,2].unsqueeze(1)
    d = rot[:,3].unsqueeze(1)

    row_1_1 = 1 - 2*(c**2 + d**2)
    row_1_2 = 2*(b*c - d*a)
    row_1_3 = 2*(b*d + c*a)

    row_2_1 = 2*(b*c + d*a)
    row_2_2 = 1 - 2*(b**2 + d**2)
    row_2_3 = 2*(c*d - b*a)

    row_3_1 = 2*(b*d - c*a)
    row_3_2 = 2*(c*d + b*a)
    row_3_3 = 1 - 2*(b**2 + c**2)

    row_1 = torch.cat((row_1_1, row_1_2, row_1_3), dim=1).unsqueeze(2)
    row_2 = torch.cat((row_2_1, row_2_2, row_2_3), dim=1).unsqueeze(2)
    row_3 = torch.cat((row_3_1, row_3_2, row_3_3), dim=1).unsqueeze(2)
    R = torch.cat((row_1, row_2, row_3), dim=2)
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
            which are the refinable parameters)
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
    # Translate the sites.
    fractional_coords = rot_cart + translation.view(translation.shape[0], 1, 3)
    return fractional_coords

def get_asymmetric_coords(external, internal, position, rotation, torsion,
    initial_D2, zmatrices_degrees_of_freedom, bond_connection, angle_connection,
    torsion_connection, torsion_refinable_indices, lattice_inv_matrix,
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
        torsion_refinable_indices (List): List of tensors that contain the
            indices of the refinable torsions for each zmatrix
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
                                        torsion_refinable_indices[i])
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
    if n_samples is None:
        n_samples = 1

    # Load the tensors needed
    tensors = tensor_prep.get_all_required_tensors(
                    Structure, external=external, internal=internal,
                    n_samples=n_samples, device=device, dtype=dtype,
                    verbose=verbose)

    asymmetric_frac_coords = get_asymmetric_coords(**tensors["zm"])


    return asymmetric_frac_coords.detach().cpu().numpy()

##
# TODO:
# - Update function inputs to accept ["zm"] tensors
# - Fix issue with lists and type checking if JIT is used
##
def zm_to_cart_writer(zm_number, samples, atoms, init_D2, bond_connect, angle_connect, torsion_connect, refinable_torsion_indices, usejit=True):
    lines_D2 = ["def modify_D2_"+str(zm_number)+"(tors, D2, refinable_torsion_indices):\n"
            "\tcos_phi = torch.cos(tors)",
            "\tsin_phi = torch.sin(tors)",
            "\tnew_D2 = D2.clone()",
            "\tnew_D2[:,refinable_torsion_indices,1] = torch.mul(D2[:,refinable_torsion_indices,1], cos_phi)",
            "\tnew_D2[:,refinable_torsion_indices,2] = torch.mul(D2[:,refinable_torsion_indices,2], sin_phi)",
            "\treturn new_D2\n"]
    lines_zm =["def unroll_zm_to_cart"+str(zm_number)+"(torsions, init_D2, refinable_torsion_indices):",
            "\t# type: (Tensor, Tensor, Tensor) -> Tensor",
            "\tD2 = modify_D2_"+str(zm_number)+"(torsions, init_D2, refinable_torsion_indices)"]
    if usejit:
        print("Using JIT for ZM", zm_number)
        lines = ["@torch.jit.script"] + lines_D2 + ["@torch.jit.script"] + lines_zm
    else:
        lines = lines_D2 + lines_zm

    for i in range(atoms):
        if i < 3:
            lines.append("\ta"+str(i)+" = D2[:,"+str(i)+",:]")
        else:
            lines.append("\t#Atom"+str(i))
            lines.append("\tC"+str(i)+" = a"+str(bond_connect[i]))
            lines.append("\tB"+str(i)+" = a"+str(angle_connect[i]))
            lines.append("\tA"+str(i)+" = a"+str(torsion_connect[i]))

            lines.append("\tc_minus_b"+str(i)+" = C"+str(i)+" - B"+str(i))
            lines.append("\tbc"+str(i)+" = c_minus_b"+str(i)+".div(torch.norm(c_minus_b"+str(i)+",dim=-1, keepdim=True))")
            lines.append("\tAB"+str(i)+" = B"+str(i)+" - A"+str(i))
            lines.append("\tn_1"+str(i)+" = torch.cross(AB"+str(i)+", bc"+str(i)+")")
            lines.append("\tn"+str(i)+" = n_1"+str(i)+".div(torch.norm(n_1"+str(i)+",dim=-1, keepdim=True))")
            lines.append("\tn_bc"+str(i)+" = torch.cross(n"+str(i)+", bc"+str(i)+")")
            #lines.append("\tM"+str(i)+" = torch.transpose(torch.cat((bc"+str(i)+", n_bc"+str(i)+", n"+str(i)+"), dim=-1).reshape(torsions.shape[0],3,3),1,2)")
            lines.append("\tM"+str(i)+" = torch.stack((bc"+str(i)+", n_bc"+str(i)+", n"+str(i)+"), dim=-1)")
            lines.append("\ta"+str(i)+" = torch.einsum(\"bij,bj->bi\",M"+str(i)+", D2[:,"+str(i)+"]) + C"+str(i))
            #lines.append("\ta"+str(i)+" = torch.matmul(M"+str(i)+", D2[:,"+str(i)+"].view(D2.shape[0], 3, 1)).view(D2.shape[0], 3) + C"+str(i))

    lines.append("\treturn torch.stack(["+", ".join(["a"+str(i) for i in range(atoms)])+"], dim=1)\n")
    return lines


def unrolled_module_writer(input_tensors, filename="unrolled.py", usejit=True):
    position, rotation, torsion, initial_D2, zmatrices_degrees_of_freedom, \
        bond_connection, angle_connection, torsion_connection, \
        torsion_refinable_indices, init_cart_coords = input_tensors
    import os
    position = [x.cpu().numpy() for x in position]
    rotation = [x.cpu().numpy() for x in rotation]
    torsion = [x.cpu().numpy() for x in torsion]
    initial_D2 = [x.cpu().numpy() for x in initial_D2]
    bond_connection = [x.cpu().numpy() for x in bond_connection]
    angle_connection = [x.cpu().numpy() for x in angle_connection]
    torsion_connection = [x.cpu().numpy() for x in torsion_connection]
    torsion_refinable_indices = [x.cpu().numpy() for x in torsion_refinable_indices]
    init_cart_coords = [x.cpu().numpy() for x in init_cart_coords if not isinstance(x, list)]

    rotation_matrix = ["def get_rotation_matrices(rot):",
    "\ta = rot[:,0].reshape(rot.shape[0],1)",
    "\tb = rot[:,1].reshape(rot.shape[0],1)",
    "\tc = rot[:,2].reshape(rot.shape[0],1)",
    "\td = rot[:,3].reshape(rot.shape[0],1)",
    "\trow_1_1 = 1 - 2*(c*c + d*d)", # Modify c**2 to c*c and d**2 to d*d - get aten:: use for some odd reason, but this fixes it
    "\trow_1_2 = 2*(b*c - d*a)",
    "\trow_1_3 = 2*(b*d + c*a)",
    "\trow_2_1 = 2*(b*c + d*a)",
    "\trow_2_2 = 1 - 2*(b*b + d*d)",# Modify b**2 to b*b and d**2 to d*d - get aten:: use for some odd reason, but this fixes it
    "\trow_2_3 = 2*(c*d - b*a)",
    "\trow_3_1 = 2*(b*d - c*a)",
    "\trow_3_2 = 2*(c*d + b*a)",
    "\trow_3_3 = 1 - 2*(b*b + c*c)",# Modify b**2 to b*b and c**2 to c*c - get aten:: use for some odd reason, but this fixes it
    "\tR = torch.cat([row_1_1,row_1_2,row_1_3, row_2_1,row_2_2,row_2_3, row_3_1,row_3_2,row_3_3],dim=1).reshape(rot.shape[0], 3, 3)",
    "\tR = torch.transpose(R, 2, 1)",
    "\treturn R\n"]
    if usejit:
        print("Using JIT for rotation")
        rotation_matrix = ["@torch.jit.script"] + rotation_matrix

    header = "import torch\n\n"
    lines = ["def get_fractional_coords(external, internal, position, rotation, torsion, initial_D2, torsion_refinable_indices, lattice_inv_matrix, init_cart_coords):",
             "\t# type: (Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], Tensor, List[Tensor]) -> Tensor"]
    if usejit:
        lines = ["@torch.jit.script"] + lines
    zm_to_cart_lines = []
    for i, zm_dof in enumerate(zmatrices_degrees_of_freedom):
        lines.append("\t#Z-matrix "+str(i))
        lines.append("\ttranslation"+str(i)+" = external[:,position["+str(i)+"]].reshape(external.shape[0], 1, "+str(len(position[i]))+")")
        if zm_dof > 7:
            lines.append("\trot_in"+str(i)+" = external[:,rotation["+str(i)+"]]") # Quaternions for rotation
            lines.append("\trot"+str(i)+" = rot_in"+str(i)+".div(torch.norm(rot_in"+str(i)+",dim=-1,keepdim=True))") # Normalize quaternions
            lines.append("\tR"+str(i)+" = get_rotation_matrices(rot"+str(i)+")") # Convert quaternions to rotation matrices
            zm_to_cart_lines.append(zm_to_cart_writer(i, initial_D2[i].shape[0], initial_D2[i].shape[1], initial_D2[i],\
                                    bond_connection[i], angle_connection[i], torsion_connection[i], torsion_refinable_indices[i], usejit=usejit))
            lines.append("\tcart_coords"+str(i)+"= unroll_zm_to_cart"+str(i)+"(internal[:,torsion["+str(i)+"]], initial_D2["+str(i)+"], torsion_refinable_indices["+str(i)+"])")
            lines.append("\trot_cart"+str(i)+" = torch.einsum(\"bjk,bij->bik\", R"+str(i)+", cart_coords"+str(i)+")")
            lines.append("\tfrac_rot_cart"+str(i)+" = torch.einsum(\"jk,bij->bik\",lattice_inv_matrix,rot_cart"+str(i)+")")
            #lines.append("\tfractional"+str(i)+" = frac_rot_cart"+str(i)+" + torch.repeat_interleave(translation"+str(i)+", repeats=cart_coords"+str(i)+".shape[1], dim=0).reshape(translation"+str(i)+".shape[0], cart_coords"+str(i)+".shape[1], 3)")
            #lines.append("\tfractional"+str(i)+" = frac_rot_cart"+str(i)+" + translation"+str(i)+".repeat(1,1,cart_coords"+str(i)+".shape[1]).reshape(translation"+str(i)+".shape[0], cart_coords"+str(i)+".shape[1], 3)")
            lines.append("\tfractional"+str(i)+" = frac_rot_cart"+str(i)+" + translation"+str(i)+".view(translation"+str(i)+".shape[0], 1, 3)")

        elif zm_dof == 7: # Rigid bodies
            lines.append("\trot_in"+str(i)+" = external[:,rotation["+str(i)+"]]") # Quaternions for rotation
            lines.append("\trot"+str(i)+" = rot_in"+str(i)+".div(torch.norm(rot_in"+str(i)+",dim=-1,keepdim=True))") # Normalize quaternions
            lines.append("\tR"+str(i)+" = get_rotation_matrices(rot"+str(i)+")") # Convert quaternions to rotation matrices
            lines.append("\tcart_coords"+str(i)+" = init_cart_coords["+str(i)+"]")
            lines.append("\trot_cart"+str(i)+" = torch.einsum(\"bjk,bij->bik\", R"+str(i)+", cart_coords"+str(i)+")")
            lines.append("\tfrac_rot_cart"+str(i)+" = torch.einsum(\"jk,bij->bik\",lattice_inv_matrix,rot_cart"+str(i)+")")
            #lines.append("\tfractional"+str(i)+" = frac_rot_cart"+str(i)+" + torch.repeat_interleave(translation"+str(i)+", repeats=cart_coords"+str(i)+".shape[1], dim=0).reshape(translation"+str(i)+".shape[0], cart_coords"+str(i)+".shape[1], 3)")
            #lines.append("\tfractional"+str(i)+" = frac_rot_cart"+str(i)+" + translation"+str(i)+".repeat(1,1,cart_coords"+str(i)+".shape[1]).reshape(translation"+str(i)+".shape[0], cart_coords"+str(i)+".shape[1], 3)")
            lines.append("\tfractional"+str(i)+" = frac_rot_cart"+str(i)+" + translation"+str(i)+".view(translation"+str(i)+".shape[0], 1, 3)")
        else: # Single atoms
            lines.append("\tfractional"+str(i)+" = translation"+str(i))
    lines.append("\treturn torch.cat(["+", ".join(["fractional"+str(i) for i in range(len(zmatrices_degrees_of_freedom))])+"], dim=1)\n")
    if os.path.exists(filename):
        print("Deleting previous")
        os.remove(filename)
    with open(filename, "w") as out_file:
        out_file.write(header)
        for line in rotation_matrix:
            out_file.write(line+"\n")
        for zm_writer in zm_to_cart_lines:
            for line in zm_writer:
                out_file.write(line+"\n")
        for line in lines:
            out_file.write(line+"\n")
    out_file.close()
    return [header] + rotation_matrix + zm_to_cart_lines + lines


# Rewritten code using f-strings with ChatGPT

def zm_to_cart_writer(zm_number, samples, atoms, init_D2, bond_connect, angle_connect, torsion_connect, refinable_torsion_indices, usejit=True):
    lines_D2 = [f"def modify_D2_{zm_number}(tors, D2, refinable_torsion_indices):",
                f"\tcos_phi = torch.cos(tors)",
                f"\tsin_phi = torch.sin(tors)",
                f"\tnew_D2 = D2.clone()",
                f"\tnew_D2[:, refinable_torsion_indices, 1] = torch.mul(D2[:, refinable_torsion_indices, 1], cos_phi)",
                f"\tnew_D2[:, refinable_torsion_indices, 2] = torch.mul(D2[:, refinable_torsion_indices, 2], sin_phi)",
                f"\treturn new_D2"]

    lines_zm = [f"def unroll_zm_to_cart{zm_number}(torsions, init_D2, refinable_torsion_indices):",
                f"\t# type: (Tensor, Tensor, Tensor) -> Tensor",
                f"\tD2 = modify_D2_{zm_number}(torsions, init_D2, refinable_torsion_indices)"]

    if usejit:
        print(f"Using JIT for ZM {zm_number}")
        lines = [f"@torch.jit.script"] + lines_D2 + [f"@torch.jit.script"] + lines_zm
    else:
        lines = lines_D2 + lines_zm

    for i in range(atoms):
        if i < 3:
            lines.append(f"\ta{i} = D2[:, {i}, :]")
        else:
            lines.append(f"\t# Atom {i}")
            lines.append(f"\tC{i} = a{bond_connect[i]}")
            lines.append(f"\tB{i} = a{angle_connect[i]}")
            lines.append(f"\tA{i} = a{torsion_connect[i]}")
            lines.append(f"\tc_minus_b{i} = C{i} - B{i}")
            lines.append(f"\tbc{i} = c_minus_b{i}.div(torch.norm(c_minus_b{i}, dim=-1, keepdim=True))")
            lines.append(f"\tAB{i} = B{i} - A{i}")
            lines.append(f"\tn_1{i} = torch.cross(AB{i}, bc{i})")
            lines.append(f"\tn{i} = n_1{i}.div(torch.norm(n_1{i}, dim=-1, keepdim=True))")
            lines.append(f"\tn_bc{i} = torch.cross(n{i}, bc{i})")
            lines.append(f"\tM{i} = torch.stack((bc{i}, n_bc{i}, n{i}), dim=-1)")
            lines.append(f"\ta{i} = torch.einsum(\"bij,bj->bi\", M{i}, D2[:, {i}]) + C{i}")

    lines.append(f"\treturn torch.stack([a{i} for i in range(atoms)], dim=1)")

    return lines


def unrolled_module_writer(input_tensors, filename="unrolled.py", usejit=True):
    position, rotation, torsion, initial_D2, zmatrices_degrees_of_freedom, bond_connection, angle_connection, torsion_connection, torsion_refinable_indices, init_cart_coords = input_tensors
    import os

    position = [x.cpu().numpy() for x in position]
    rotation = [x.cpu().numpy() for x in rotation]
    torsion = [x.cpu().numpy() for x in torsion]
    initial_D2 = [x.cpu().numpy() for x in initial_D2]
    bond_connection = [x.cpu().numpy() for x in bond_connection]
    angle_connection = [x.cpu().numpy() for x in angle_connection]
    torsion_connection = [x.cpu().numpy() for x in torsion_connection]
    torsion_refinable_indices = [x.cpu().numpy() for x in torsion_refinable_indices]
    init_cart_coords = [x.cpu().numpy() for x in init_cart_coords if not isinstance(x, list)]

    rotation_matrix = [f"def get_rotation_matrices(rot):",
                      f"\ta = rot[:, 0].reshape(rot.shape[0], 1)",
                      f"\tb = rot[:, 1].reshape(rot.shape[0], 1)",
                      f"\tc = rot[:, 2].reshape(rot.shape[0], 1)",
                      f"\td = rot[:, 3].reshape(rot.shape[0], 1)",
                      f"\trow_1_1 = 1 - 2 * (c * c + d * d)",
                      f"\trow_1_2 = 2 * (b * c - d * a)",
                      f"\trow_1_3 = 2 * (b * d + c * a)",
                      f"\trow_2_1 = 2 * (b * c + d * a)",
                      f"\trow_2_2 = 1 - 2 * (b * b + d * d)",
                      f"\trow_2_3 = 2 * (c * d - b * a)",
                      f"\trow_3_1 = 2 * (b * d - c * a)",
                      f"\trow_3_2 = 2 * (c * d + b * a)",
                      f"\trow_3_3 = 1 - 2 * (b * b + c * c)",
                      f"\tR = torch.cat([row_1_1, row_1_2, row_1_3, row_2_1, row_2_2, row_2_3, row_3_1, row_3_2, row_3_3], dim=1).reshape(rot.shape[0], 3, 3)",
                      f"\tR = torch.transpose(R, 2, 1)",
                      f"\treturn R"]

    if usejit:
        print("Using JIT for rotation")
        rotation_matrix = [f"@torch.jit.script"] + rotation_matrix

    header = "import torch\n\n"
    lines = [f"def get_fractional_coords(external, internal, position, rotation, torsion, initial_D2, torsion_refinable_indices, lattice_inv_matrix, init_cart_coords):",
             f"\t# type: (Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], Tensor, List[Tensor]) -> Tensor"]

    if usejit:
        lines = [f"@torch.jit.script"] + lines

    zm_to_cart_lines = []

    for i, zm_dof in enumerate(zmatrices_degrees_of_freedom):
        lines.append(f"\t# Z-matrix {i}")
        lines.append(f"\ttranslation{i} = external[:, position[{i}]].reshape(external.shape[0], 1, {len(position[i])})")

        if zm_dof > 7:
            lines.append(f"\trot_in{i} = external[:, rotation[{i}]]")

  # Quaternions for rotation
            lines.append(f"\trot{i} = rot_in{i}.div(torch.norm(rot_in{i}, dim=-1, keepdim=True))")  # Normalize quaternions
            lines.append(f"\tR{i} = get_rotation_matrices(rot{i})")  # Convert quaternions to rotation matrices

            zm_to_cart_lines.extend(zm_to_cart_writer(i, initial_D2[i].shape[0], initial_D2[i].shape[1], initial_D2[i],
                                                      bond_connection[i], angle_connection[i], torsion_connection[i],
                                                      torsion_refinable_indices[i], usejit=usejit))

            lines.append(f"\tcart_coords{i} = unroll_zm_to_cart{i}(internal[:, torsion[{i}]], initial_D2[{i}], torsion_refinable_indices[{i}])")
            lines.append(f"\trot_cart{i} = torch.einsum(\"bjk,bij->bik\", R{i}, cart_coords{i})")
            lines.append(f"\tfrac_rot_cart{i} = torch.einsum(\"jk,bij->bik\", lattice_inv_matrix, rot_cart{i})")
            lines.append(f"\tfractional{i} = frac_rot_cart{i} + translation{i}.view(translation{i}.shape[0], 1, 3)")

        elif zm_dof == 7:  # Rigid bodies
            lines.append(f"\trot_in{i} = external[:, rotation[{i}]]")  # Quaternions for rotation
            lines.append(f"\trot{i} = rot_in{i}.div(torch.norm(rot_in{i}, dim=-1, keepdim=True))")  # Normalize quaternions
            lines.append(f"\tR{i} = get_rotation_matrices(rot{i})")  # Convert quaternions to rotation matrices
            lines.append(f"\tcart_coords{i} = init_cart_coords[{i}]")
            lines.append(f"\trot_cart{i} = torch.einsum(\"bjk,bij->bik\", R{i}, cart_coords{i})")
            lines.append(f"\tfrac_rot_cart{i} = torch.einsum(\"jk,bij->bik\", lattice_inv_matrix, rot_cart{i})")
            lines.append(f"\tfractional{i} = frac_rot_cart{i} + translation{i}.view(translation{i}.shape[0], 1, 3)")

        else:  # Single atoms
            lines.append(f"\tfractional{i} = translation{i}")

    lines.append(f"\treturn torch.cat([fractional{i} for i in range(len(zmatrices_degrees_of_freedom))], dim=1)")

    if os.path.exists(filename):
        print("Deleting previous")
        os.remove(filename)

    with open(filename, "w") as out_file:
        out_file.write(header)
        for line in rotation_matrix:
            out_file.write(line + "\n")
        for zm_writer in zm_to_cart_lines:
            for line in zm_writer:
                out_file.write(line + "\n")
        for line in lines:
            out_file.write(line + "\n")

    return [header] + rotation_matrix + zm_to_cart_lines + lines
