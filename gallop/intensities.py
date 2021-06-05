# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Calculates PXRD intensities given fractional atomic coordinates.
"""

import torch

# Not PEP8 compliant in most of the file but it's been formatted to try to be
# readable

@torch.jit.script
def get_symmetry_equivalent_points(frac_coords, nsamples_ones, affine_matrices):
    """
    This code is a generic way to find all symmetry related positions in the
    unit cell from the affine-matrices of the space group. There are faster ways
    to generate intensities for some space groups, but these need to be coded
    separately. This function works in the generic case.

    Args:
        frac_coords (Tensor): Fractional coordinates for the asymmetric unit,
            with shape N_samples, Number of atoms in asymmetric unit, 3
        nsamples_ones (Tensor): A tensor filled with 1s, of shape:
            N_samples, Number of atoms in asymmetric unit, 1
        affine_matrices (Tensor): The affine matrices for the space group

    Returns:
        Tensor: The coordinates of all atoms in the unit cell. Tensor with shape
            N_samples, Number of atoms in unit cell, 3
    """

    wxyz = torch.cat((frac_coords, nsamples_ones), dim=-1)
    frac_all = torch.einsum("bij,nkj->nbik",wxyz,affine_matrices)[:,:,:,:3]
    n_samples = frac_coords.shape[0]
    n_atoms = frac_coords.shape[1]*affine_matrices.shape[0]
    return frac_all.permute(1,0,2,3).reshape(n_samples,n_atoms,3)

@torch.jit.script
def intensities_from_full_cell_contents(frac_all, hkl, intensity_calc_prefix_fs,
    centrosymmetric):
    # type: (Tensor, Tensor, Tensor, bool) -> Tensor
    """
    Fall back generic method for intensity calculation if the space group is not
    included in calculate_intensities

    Args:
        frac_all (Tensor): Coordinates of all atoms in the unit cell
        hkl (Tensor): Miller indices
        intensity_calc_prefix_fs (Tensor): atomic scattering factors etc
        centrosymmetric (bool): True if space group is centrosymmetric

    Returns:
        [type]: [description]
    """
    pi = 3.141592653589793
    hxkylz = 2. * pi * torch.einsum("ji,klj->kil", hkl, frac_all)
    Asqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs,torch.cos(hxkylz)).sum(dim=2)**2
    if centrosymmetric:
        return Asqd
    else:
        Bsqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs,torch.sin(hxkylz)).sum(dim=2)**2
        return Asqd + Bsqd

@torch.jit.script
def calculate_intensities(asymmetric_frac_coords, hkl, intensity_calc_prefix_fs,
        intensity_calc_prefix_fs_asymmetric, nsamples_ones, affine_matrices,
        centrosymmetric, space_group_number):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, int) -> Tensor
    """
    Calculate the intensities for a set of Miller indices and atomic fractional
    coordinates. For some space groups, an equation taken from the International
    Tables of Crystallography has been used to allow the structure factors to be
    determined directly from the contents of the asymmetric unit.
    If this equation has not been added to the code, then the fall-back method
    is to generate the equivalent positions within the unit cell, and then
    determine the structure factors using the general P1 equation (or P-1 if
    centrosymmetric). Note: equations taken from international tables may not
    be faster or use less memory than the fallback method - testing is advised!

    Args:
        asymmetric_frac_coords (Tensor): The fractional coordinates of the atoms
            in the asymmetric unit
        hkl (Tensor): The hkl Miller indices of the observed reflections
        intensity_calc_prefix_fs (Tensor): The atomic scattering factors,
            occupancies and DW-factors of all atoms in the unit cell.
            Only used in fallback method, generated with a Structure object.
        intensity_calc_prefix_fs_asymmetric (Tensor): The atomic scattering
            factors, occupancies and DW-factors of atoms in the asymmetric unit.
            Generated with a Structure object.
        nsamples_ones (Tensor): a tensor of ones, with the
            shape(nsamples, natoms, 1). Only used in fallback method
        affine_matrices (Tensor): Tensor of affine matrices obtained from
            pymatgen. Only used in fallback method
        centrosymmetric (bool): True if the space group is centrosymmetric
            else False
        space_group_number (integer): The space group number for the current
            crystal

    Returns:
        intensities: A tensor of shape (n_samples, n_reflections) containing
                    the calculated intensities for all hkl's.
    """
    pi = 3.141592653589793
    peaks = hkl.shape[1]
    if space_group_number == 1:
        # P1
        hxkylz = 2 * pi * torch.einsum("ji,klj->kil", hkl, asymmetric_frac_coords)
        Asqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,torch.cos(hxkylz)).sum(dim=2)**2
        Bsqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,torch.sin(hxkylz)).sum(dim=2)**2
        # (a + ib) * (a - ib) = a^2 + b^2
        intensities = Asqd + Bsqd

    elif space_group_number == 2:
        # P-1
        hxkylz = 2 * pi * torch.einsum("ji,klj->kil", hkl, asymmetric_frac_coords)
        A = 2 * torch.cos(hxkylz)
        intensities = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2

    elif space_group_number == 3:
        # P2
        hl = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        + torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2]))
        ky = 2 * pi * torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])

        chl = torch.cos(hl)
        cky = torch.cos(ky)
        sky = torch.sin(ky)

        A = 2 * chl * cky
        B = 2 * chl * sky
        Asqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2
        Bsqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,B).sum(dim=2)**2
        intensities = Asqd + Bsqd

    elif space_group_number == 4:
        # P21
        hl = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        + torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                        + hkl[1].view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        - hkl[1].view(1,peaks,1)/4)

        chl = torch.cos(hl)
        cky = torch.cos(ky)
        sky = torch.sin(ky)

        A = 2 * chl * cky
        B = 2 * chl * sky

        Asqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2
        Bsqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,B).sum(dim=2)**2
        intensities = Asqd + Bsqd

    elif space_group_number == 5:
        # C2
        # This expression is simpler and quicker than the style used
        # in the other space groups. For reference, it would be:
        # A = 4 * cos(2pi (h+k/4))*cos(2pi(hx+lz))*cos(2piky)
        # B = 4 * cos(2pi (h+k/4))*cos(2pi(hx+lz))*sin(2piky)
        # This would require a minimum of 3 x trig functions to determine
        # whereas the following only requires 2.
        chl = torch.cos(2.*pi*(torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                                + torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])))
        ky = 2 * pi * torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])

        A = 4 * chl * torch.cos(ky)
        B = 4 * chl * torch.sin(ky)

        Asqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2
        Bsqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,B).sum(dim=2)**2
        intensities = Asqd + Bsqd

    elif space_group_number == 7:
        # Pc
        hl = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        + torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                        + hkl[2].view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        - hkl[2].view(1,peaks,1)/4)

        chl = torch.cos(hl)
        shl = torch.sin(hl)
        cky = torch.cos(ky)
        A = 2 * chl * cky
        B = 2 * shl * cky

        Asqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2
        Bsqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,B).sum(dim=2)**2
        intensities = Asqd + Bsqd

    elif space_group_number == 9:
        # Cc
        hl = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        + torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                        + hkl[2].view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        - hkl[2].view(1,peaks,1)/4)
        hk = (2 * pi * (hkl[0] + hkl[1])/4).view(1,peaks,1)

        chl = torch.cos(hl)
        shl = torch.sin(hl)
        cky = torch.cos(ky)
        c_sqd_hk = torch.cos(hk)**2
        A = 4 * c_sqd_hk * chl * cky
        B = 4 * c_sqd_hk * shl * cky

        Asqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2
        Bsqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,B).sum(dim=2)**2
        intensities = Asqd + Bsqd

    elif space_group_number == 11:
        # P21/m
        hl = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        + torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                        + hkl[1].view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        - hkl[1].view(1,peaks,1)/4)

        A = 4 * torch.cos(hl) * torch.cos(ky)
        intensities = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2

    elif space_group_number == 12:
        # C2/m
        hl = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        + torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2]))
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1]))
        hk = (2 * pi * (hkl[0] + hkl[1])/4).view(1,peaks,1)
        c_sqd_hk = torch.cos(hk)**2
        A = 8 * c_sqd_hk * torch.cos(hl) * torch.cos(ky)
        intensities = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2

    elif space_group_number == 13:
        # P2/c
        hl = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        + torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                        + hkl[2].view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        - hkl[2].view(1,peaks,1)/4)
        A = 4 * torch.cos(hl) * torch.cos(ky)
        intensities = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2

    elif space_group_number == 14:
        # P21/c
        hl = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        + torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                        + (hkl[1] + hkl[2]).view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        - (hkl[1] + hkl[2]).view(1,peaks,1)/4)
        A = 4 * torch.cos(hl) * torch.cos(ky)
        intensities = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2

    elif space_group_number == 15:
        # C2/c
        # The initial term involving only h and k could be calculated in advance.
        hl = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        + torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                        + hkl[2].view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        - hkl[2].view(1,peaks,1)/4)

        A = 8 * (((torch.cos((2 * pi * (hkl[0] + hkl[1]))/4))**2).view(1,peaks,1)
                    * torch.cos(hl) * torch.cos(ky))

        intensities = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2

    elif space_group_number == 18:
        # P21212
        hx = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        + (hkl[0] + hkl[1]).view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        - (hkl[0] + hkl[1]).view(1,peaks,1)/4)
        lz = 2 * pi * torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])

        chx = torch.cos(hx)
        cky = torch.cos(ky)
        clz = torch.cos(lz)

        shx = torch.sin(hx)
        sky = torch.sin(ky)
        slz = torch.sin(lz)

        A =  4 * chx * cky * clz
        B = -4 * shx * sky * slz

        Asqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2
        Bsqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,B).sum(dim=2)**2
        intensities = Asqd + Bsqd

    elif space_group_number == 19:
        # P212121
        hx = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        - (hkl[0] - hkl[1]).view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        - (hkl[1] - hkl[2]).view(1,peaks,1)/4)
        lz = 2 * pi * (torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                        - (hkl[2] - hkl[0]).view(1,peaks,1)/4)

        chx = torch.cos(hx)
        cky = torch.cos(ky)
        clz = torch.cos(lz)

        shx = torch.sin(hx)
        sky = torch.sin(ky)
        slz = torch.sin(lz)

        A =  4 * chx * cky * clz
        B = -4 * shx * sky * slz

        Asqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2
        Bsqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,B).sum(dim=2)**2
        intensities = Asqd + Bsqd

    elif space_group_number == 29:
        # Pca21
        hx = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        - (hkl[0] + hkl[2]).view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        + (hkl[0]).view(1,peaks,1)/4)
        lz = 2 * pi * (torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                        + (hkl[2]).view(1,peaks,1)/4)
        chx = torch.cos(hx)
        cky = torch.cos(ky)
        clz = torch.cos(lz)
        slz = torch.sin(lz)

        A = 4 * chx * cky * clz
        B = 4 * chx * cky * slz

        Asqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2
        Bsqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,B).sum(dim=2)**2
        intensities = Asqd + Bsqd

    elif space_group_number == 33:
        # Pna21
        hx = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        - (hkl[0] + hkl[1] + hkl[2]).view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        + (hkl[0] + hkl[1]).view(1,peaks,1)/4)
        lz = 2 * pi * (torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                        + (hkl[2]).view(1,peaks,1)/4)
        chx = torch.cos(hx)
        cky = torch.cos(ky)
        clz = torch.cos(lz)
        slz = torch.sin(lz)

        A = 4 * chx * cky * clz
        B = 4 * chx * cky * slz

        Asqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2
        Bsqd = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,B).sum(dim=2)**2
        intensities = Asqd + Bsqd

    elif space_group_number == 60:
        # Pbcn
        hx = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        + (hkl[0] + hkl[1]).view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        + (hkl[2]).view(1,peaks,1)/4)
        lz = 2 * pi * (torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                        - (hkl[0] + hkl[1] + hkl[2]).view(1,peaks,1)/4)
        chx = torch.cos(hx)
        cky = torch.cos(ky)
        clz = torch.cos(lz)

        A = 8 * chx * cky * clz
        intensities = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2

    elif space_group_number == 61:
        # Pbca
        hx = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        - (hkl[0] - hkl[1]).view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        - (hkl[1] - hkl[2]).view(1,peaks,1)/4)
        lz = 2 * pi * (torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                        - (hkl[2] - hkl[0]).view(1,peaks,1)/4)
        chx = torch.cos(hx)
        cky = torch.cos(ky)
        clz = torch.cos(lz)

        A = 8 * chx * cky * clz
        intensities = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2

    elif space_group_number == 62:
        # Pnma
        hx = 2 * pi * (torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                        - (hkl[0] + hkl[1] + hkl[2]).view(1,peaks,1)/4)
        ky = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                        + (hkl[1]).view(1,peaks,1)/4)
        lz = 2 * pi * (torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                        + (hkl[0] + hkl[2]).view(1,peaks,1)/4)
        chx = torch.cos(hx)
        cky = torch.cos(ky)
        clz = torch.cos(lz)

        A = 8 * chx * cky * clz
        intensities = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2

    elif space_group_number == 88:
        # I41/a
        chkl  = torch.cos(2*pi*(hkl[0] + hkl[1] + hkl[2]).view(1,peaks,1)/4)
        chxky = torch.cos(2*pi*(torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,0])
                                + torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,1])
                                - (hkl[1]).view(1,peaks,1)/4))
        clz_k = torch.cos(2*pi*(torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                                + (hkl[1]).view(1,peaks,1)/4))
        chykx = torch.cos(2*pi*(torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,1])
                                - torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,0])
                                - (hkl[0]).view(1,peaks,1)/4))
        clz_h = torch.cos(2*pi*(torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2])
                                + (hkl[0]).view(1,peaks,1)/4))

        A = 8 * chkl * ((chkl * chxky * clz_k) + (chykx * clz_h))
        intensities = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2

    elif space_group_number == 148:
        # R-3
        # Using Rhombohedral coordinates
        hxkylz = 2 * pi * torch.einsum("ji,klj->kil", hkl, asymmetric_frac_coords)
        kxlyhz = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,0])
                            + torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,1])
                            + torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,2]))
        lxhykz = 2 * pi * (torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,0])
                            + torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,1])
                            + torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,2]))

        chx = torch.cos(hxkylz)
        ckx = torch.cos(kxlyhz)
        clx = torch.cos(lxhykz)

        A = 2 * (chx + ckx + clx)

        # Using hexagonal coordinates, the equivalent equations would be as below
        #i = -1*(hkl[0] + hkl[1])
        #chkl   = torch.cos(2 * pi * (hkl[1] + hkl[2] - hkl[0]).view(1,peaks,1)/3)
        #hxkylz = 2 * pi * torch.einsum("ji,klj->kil", hkl, asymmetric_frac_coords)
        #kxiyhz = 2 * pi * (torch.einsum("i,jk->jik", hkl[1], asymmetric_frac_coords[:,:,0])
        #                   + torch.einsum("i,jk->jik", i, asymmetric_frac_coords[:,:,1])
        #                   + torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2]))
        #ixhylz = 2 * pi * (torch.einsum("i,jk->jik", i, asymmetric_frac_coords[:,:,0])
        #                   + torch.einsum("i,jk->jik", hkl[0], asymmetric_frac_coords[:,:,1])
        #                   + torch.einsum("i,jk->jik", hkl[2], asymmetric_frac_coords[:,:,2]))
        #chx = torch.cos(hxkylz)
        #ckx = torch.cos(kxiyhz)
        #cix = torch.cos(ixhylz)#
        #A = 2 * (1 + 2*chkl) * chx * ckx * cix

        intensities = torch.einsum("ij,bij->bij",intensity_calc_prefix_fs_asymmetric,A).sum(dim=2)**2

    else:
        # Use a fall-back method to generate all equivalent positions in the unit cell
        frac_all = get_symmetry_equivalent_points(asymmetric_frac_coords,
                                    nsamples_ones, affine_matrices)
        # And then the P1 structure factor equation to obtain the intensities.
        intensities = intensities_from_full_cell_contents(frac_all, hkl,
                                    intensity_calc_prefix_fs, centrosymmetric)

    return intensities

@torch.jit.script
def apply_MD_PO_correction(calculated_intensities, cosP, sinP, factor):
    """
    An implementation of March-Dollase preferred orientation correction, adapted
    from the GSAS-II source code.

    Original code can be found in:
    GSASIIstrMath.py > GetPrefOri

    For original source code, see here:
        https://subversion.xray.aps.anl.gov/pyGSAS/trunk/GSASIIstrMath.py

    Args:
        calculated_intensities (Tensor): Calculated intensities
        cosP (Tensor): cosP, calculated in advance - see tensor_prep.py
        sinP (Tensor): sinP, calculated in advance - see tensor_prep.py
        factor (Tensor): The March-Dollase factor

    Returns:
        Tensor: Modified intensities, with PO correction applied
    """
    A_all = (1.0/torch.sqrt(((factor**2)*cosP)**2+sinP**2/(factor**2)))**3
    return calculated_intensities * A_all
