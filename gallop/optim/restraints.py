# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Calculates additional (weighted) penalties for distances, angles and torsions.
"""
import torch
from torch import Tensor

@torch.jit.script
def get_restraint_penalties(asymmetric_frac_coords, lattice_matrix, d_atoms,
                    distances, d_weights, a_atoms, cos_angles, a_weights,
                    t_atoms, sintorsions, costorsions, t_weights, restrain_d,
                    restrain_a, restrain_t):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool) -> Tensor
    """
    Supply the asymmetric fractional coordinates and atoms associated with
        pre-defined distances, angles and torsions then calculate these values
        for the atom positions in cartesian coordinates. Assign a penalty to
        those atoms where they deviate from the supplied distances, angles and
        torsions.

    Args:
        asymmetric_frac_coords (Tensor): The asymmetric unit fractional coordinates.
        lattice_matrix (Tensor): Matrix representation of the lattice used to
            convert fractional to cartesian coordinates.
        d_atoms (Tensor - long): Indices of the atoms involved in the distance
            restraints
        distances (Tensor): Expected distances
        d_weights (Tensor): Weights assigned to the distance penalty
        a_atoms (Tensor - long): Indices of the atoms involved in the angle
            restraints
        cos_angles (Tensor): Cosine of the angles used in the restraints
        a_weights (Tensor): Weights assigned to the angle penalty
        t_atoms (Tensor): Indices if the atoms involved in the torsion
            restraints
        sintorsions (Tensor): Sines of the torsion angles
        costorsions (Tensor): Cosines of the torsions angles
        t_weights   (Tensor): Weights assigned to the torsion penalty
        restrain_d (bool): Use distance restraints
        restrain_a (bool): Use angle restraints
        restrain_t (bool): Use torsion restraints

    Returns:
        Tensor: Tensor containing the weighted penalties
    """
    cart = torch.einsum("jk,bij->bik",
                    lattice_matrix,asymmetric_frac_coords)
    penalty = torch.zeros_like(asymmetric_frac_coords[:,0,0])

    if restrain_d:
        # Distances
        atomic_distances = torch.sqrt((
                (cart[:,d_atoms,:][:,:,0,:] - cart[:,d_atoms,:][:,:,1,:])**2
                    ).sum(dim=-1))
        distance_penalty = (d_weights.view(1,d_weights.shape[0])*(
                                                distances-atomic_distances)**2)
        penalty += distance_penalty.sum(dim=-1)

    if restrain_a:
        # Angles
        u = cart[:,a_atoms,:][:,:,0,:] - cart[:,a_atoms,:][:,:,1,:]
        v = cart[:,a_atoms,:][:,:,2,:] - cart[:,a_atoms,:][:,:,1,:]
        atomic_cos_angles = torch.einsum('bij,bij->bi', u, v).div(
                                u.norm(p=2, dim=-1)*v.norm(p=2, dim=-1))

        angle_penalty = (a_weights.view(1,a_weights.shape[0])*(
                                            cos_angles-atomic_cos_angles)**2)
        penalty += angle_penalty.sum(dim=-1)

    if restrain_t:
        # Torsions
        u1 = cart[:,t_atoms,:][:,:,1,:] - cart[:,t_atoms,:][:,:,0,:]
        u2 = cart[:,t_atoms,:][:,:,2,:] - cart[:,t_atoms,:][:,:,1,:]
        u3 = cart[:,t_atoms,:][:,:,3,:] - cart[:,t_atoms,:][:,:,2,:]

        u1u2 = torch.cross(u1,u2)
        u2u3 = torch.cross(u2,u3)
        u1u2n_u2u3n = u1u2.norm(p=2, dim=-1) * u2u3.norm(p=2, dim=-1)

        sinphi = u2.norm(p=2, dim=-1)*torch.einsum("bij,bij->bi", u1, u2u3).div(
                                                                    u1u2n_u2u3n)
        cosphi = torch.einsum("bij,bij->bi",u1u2, u2u3).div(u1u2n_u2u3n)

        torsion_penalty = (t_weights.view(1,t_weights.shape[0])*(
                            ((sintorsions-sinphi)**2)+(costorsions-cosphi)**2))
        penalty += torsion_penalty.sum(dim=-1)

    return penalty