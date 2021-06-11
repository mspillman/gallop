# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
GALLOP Structure class, used to hold all of the information need for GALLOP SDPD
attempts.
"""

import json
import os
import numpy as np
import pymatgen as pmg
from pymatgen.symmetry import groups
import gallop.z_matrix as zm
import gallop.files as files

class Structure(object):
    """
    The main class used in GALLOP to hold all of the information
    about a crystal structure that is needed for SDPD attempts.

    Before SDPD can be attempted, the following numpy arrays are
    needed:
        - hkl = Structure.hkl
        - intensities = Structure.intensities
        - inverse_covariance_matrix = structure.inverse_covariance_matrix
    As well as:
        - zmatrices populated with GALLOP Z_matrix objects all of
            the ZMs in the asymmetric unit
            Stored in Structure.zmatrices (list of Z-matrix objects)
        - Structure.lattice = pymatgen lattice
        - Structure.space_group = pymatgen space_group
        - Structure.sg_number = sg_number (integer)
        - Structure.centrosymmetric = centrosymmetric (bool)

    If using files generated by Pawley fitting the data with DASH, GSAS-II or
    TOPAS, helper functions are available to automate the population of
    the required arrays and parameters.

    """
    def __init__(self, name="Gallop_structure", ignore_H_atoms=True,
                    absorb_H_Z_increase=False, absorb_H_occu_increase=False):
        """
        Args:
            name (str, optional): The root name to be used for
                writing CIFs during SDPD attempts. Defaults to
                "Gallop_structure".
            ignore_H_atoms (bool, optional): Ignore H-atoms during
                SDPD. Setting this to True, and
                absorb_H_occu_increase and absorb_H_Z_increase to
                False will simply ignore the presence of H atoms.
                Defaults to True.
            absorb_H_Z_increase (bool, optional): If ignore_H_atoms,
                offset the loss of electrons in unit cell by
                increasing the atomic number of the non-H atoms by
                the number of connected H atoms. Defaults to False.
            absorb_H_occu_increase (bool, optional): If ignore_H_atoms,
                offset the loss of electrons in the unit cell by
                increasing the occupancy of the non-H atoms by
                a factor of (1 + n_H_connected_expanded)/non_H(Z).
                Defaults to False.
        """
        self.name = name
        self.ignore_H_atoms = ignore_H_atoms
        self.output_filename_root = name
        self.zmatrices = []
        self.absorb_H_Z_increase = absorb_H_Z_increase
        self.absorb_H_occu_increase = absorb_H_occu_increase

        self.original_sg_number = None
        self.total_dof_calculated = False
        self.sg_number = None
        self.wavelength = None
        self.hkl = None
        self.twotheta = []
        self.space_group = None
        self.lattice = None
        self.cif_frac_coords_no_H = None
        self.cif_species_no_H = None
        self.cif_frac_coords = None
        self.cif_species = None
        self.data_file = None
        self.dspacing = None
        self.data_resolution = None
        self.source = None
        self.profile = []
        self.baseline_peaks = []
        self.n_contributing_peaks = []
        self.PawleyChiSq = None
        self.centrosymmetric = None
        self.affine_matrices = None
        self.cif_dwfactors = None
        self.dwfactors_asymmetric = None
        self.dwfactors = None
        self.total_degrees_of_freedom = None
        self.total_external_degrees_of_freedom = None
        self.total_internal_degrees_of_freedom = None
        self.total_position_degrees_of_freedom = None
        self.total_rotation_degrees_of_freedom = None
        self.zm_torsions = None
        self.restraints = []
        self.restraints_no_H = []

    def __repr__(self):
        return self.name

    def to_json(self, return_json=True):
        """
        Save a structure object to a JSON formatted string. Can then be used to
        instantiate a structure object from a string (or file)

        Args:
            return_json (bool, optional): If True, return a JSON formatted
                string, else return a dictionary. Default is True.

        Returns:
            dict or str: JSON-compatible dict or JSON formatted string
        """
        dumpable = {}
        for k, v in self.__dict__.items():
            if k in ["zmatrices", "lattice", "space_group"]:
                if k == "zmatrices":
                    dumpable[k] = [zm.to_json(return_json=return_json) for zm in v]
                elif k == "lattice":
                    dumpable[k] = v.as_dict()
            else:
                if isinstance(v, np.ndarray):
                    dumpable[k] = [v.tolist(), True]
                elif isinstance(v, np.int32):
                    dumpable[k] = [int(v), False]
                else:
                    dumpable[k] = [v, False]
        if return_json:
            return json.dumps(dumpable)
        else:
            return dumpable

    def from_json(self, attributes, is_json=True):
        """
        Load a structure object from JSON formatted string

        Args:
            attributes (str or dict): JSON formatted string of a structure
                                    produced by the to_json method
            is_json (bool, optional): Toggles reading dict directly or using a
                JSON formatted string. Default is True
        """
        if is_json:
            attributes = json.loads(attributes)
        for k, v in attributes.items():
            if k in ["zmatrices", "lattice", "space_group"]:
                if k == "zmatrices":
                    zmatrices = []
                    for zmstring in v:
                        zmat = zm.Z_matrix()
                        zmat.from_json(zmstring, is_json=is_json)
                        zmatrices.append(zmat)
                    setattr(self, k, zmatrices)
                elif k == "lattice":
                    lattice = pmg.Lattice.from_dict(v)
                    setattr(self,k,lattice)
            else:
                if v[1]:
                    setattr(self, k, np.array(v[0]))
                else:
                    setattr(self, k, v[0])
        setattr(self, "space_group",
                        groups.SpaceGroup.from_int_number(self.sg_number))

    def add_zmatrix(self, filename, verbose=True):
        """
        Create and add a z-matrix object to the Structure

        Args:
            filename (str): filename of the Z-matrix
            verbose (bool, optional): Print out information. Defaults to True.
        """
        self.zmatrices.append(zm.Z_matrix(filename))
        if verbose:
            print("Added Z-matrix with",self.zmatrices[-1])

    def get_resolution(self, twotheta, decimal_places=3):
        """
        Calculate the d-spacing of a given twotheta value using Bragg's law

        Args:
            twotheta (float): The twotheta value of interest
            decimal_places (int, optional): Number of decimal_places to restrict
                the result to. Defaults to 3.

        Returns:
            float: the d-spacing of a twotheta value
        """
        d = self.wavelength / (2*np.sin(np.deg2rad(twotheta/2)))
        return (np.around(d, decimal_places))

    def add_data(self, filename, source="DASH",
                percentage_cutoff=20):
        """
        Add PXRD data to the Structure object

        Args:
            filename (str): filename of file from which to read the data
            source (str, optional): data source. Currently only "DASH", "GSAS"
                or "TOPAS" are accepted as an argument.
                More programs may be added in the future. Defaults to "DASH".
            percentage_cutoff (int, optional): the minimum percentage
                correlation to be included in the inverse covariance
                matrix. Defaults to 20 to be comparable with DASH,
                however, this doesn't affect the speed of GALLOP so can
                be set as desired without impacting performance.
        """
        self.data_file = filename
        if source.lower() == "dash":
            data = files.dash.get_data_from_DASH_sdi(filename,
                        percentage_cutoff_inv_cov=percentage_cutoff)
            for k, v in data.items():
                setattr(self, k, v)
            self.dspacing = self.get_resolution(self.twotheta)
            self.data_resolution = self.get_resolution(self.twotheta[-1])
            self.source = "dash"
        elif "gsas" in source.lower():
            data = files.gsas.get_data_from_GSAS_gpx(filename,
                            percentage_cutoff_inv_cov=percentage_cutoff)
            for k, v in data.items():
                setattr(self, k, v)
            self.data_resolution = self.dspacing[-1]
            self.source = "gsas"
        elif "topas" == source.lower():
            data = files.topas.get_data_from_TOPAS_output(filename,
                            percentage_cutoff_inv_cov=percentage_cutoff)
            for k, v in data.items():
                setattr(self, k, v)
            self.source = "topas"
        else:
            print("This program is not yet supported.")
            print("Currently supported programs:")
            print(" - DASH")
            print(" - GSAS-II")
            print(" - TOPAS")
        self.centrosymmetric = self.check_centre_of_symmetry()
        self.affine_matrices = self.get_affine_matrices()

    def check_centre_of_symmetry(self):
        """
        Check if a structure is centrosymmetric

        Returns:
            bool: True if centrosymmetric
        """
        laue = ["-1", "2/m", "mmm", "4/m", "4/mmm",
                    "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m"]
        return self.space_group.point_group in laue

    def get_affine_matrices(self):
        """
        Get the affine matrices of the space group
        Used as a fall-back (generic) method for structure factor
        calculation.

        GALLOP Structure must have a pymatgen space_group assigned

        Returns:
            numpy array:    affine matrices for the space group
                            of shape (n,4,4) where n is the number
                            of equivalent positions for the space
                            group.
        """
        affine_matrices = []
        for op in self.space_group.symmetry_ops:
            affine_matrices.append(op.affine_matrix)
        return np.array(affine_matrices)

    def generate_intensity_calculation_prefix(self,
                                            debye_waller_factors=None,
                                            just_asymmetric=False,
                                            from_cif=False):
        """
        Much of this code is directly adapted from the PyMatGen code
        for PXRD pattern generation, which can be found here:
        https://pymatgen.org/pymatgen.analysis.diffraction.xrd.html
        A few modifications have been made. For example, ability to "absorb"
        H atoms by increasing the atomic number used to calculate
        atomic form factors, or increase the occupancy of the non-H
        atoms.

        If reading directly from a CIF then the atomic coordinates
        are used. If not reading from a CIF, then a dummy structure
        is created in order to extract the required information.
        This can either just be the atoms in the asymmetric unit,
        or the whole unit cell.
        """
        if debye_waller_factors is None:
            debye_waller_factors = {}
        if len(self.zmatrices) == 0 and not from_cif:
            print("No Z-matrices have been added!")
        else:
            if not from_cif:
                all_atoms_coords = []
                all_atoms_elements = []
                all_atoms_n_H_connected = []
                for zmat in self.zmatrices:
                    if self.ignore_H_atoms:
                        all_atoms_coords.append(
                            zmat.initial_cartesian_no_H)
                        all_atoms_elements.append(
                            zmat.elements_no_H)
                        all_atoms_n_H_connected.append(
                            zmat.n_H_connected)
                    else:
                        all_atoms_coords.append(zmat.initial_cartesian)
                        all_atoms_elements.append(zmat.elements)

                all_atoms_coords = np.vstack(all_atoms_coords)
                all_atoms_elements = np.hstack(all_atoms_elements)
                if self.ignore_H_atoms:
                    all_atoms_n_H_connected = np.hstack(all_atoms_n_H_connected)
                else:
                    all_atoms_n_H_connected = np.array(all_atoms_n_H_connected)

                fractional_coords = np.dot(all_atoms_coords,
                                                        self.lattice.inv_matrix)
            else:
                if self.ignore_H_atoms:
                    fractional_coords = self.cif_frac_coords_no_H
                    all_atoms_elements = self.cif_species_no_H
                else:
                    fractional_coords = self.cif_frac_coords
                    all_atoms_elements = self.cif_species
                just_asymmetric = True

            if not just_asymmetric:
                # Apply symmetry of space group to locate all atoms
                # within the unit cell. This is a dummy structure,
                # and the purpose is only to extract the atomic
                # scattering parameters etc, which are independent
                # of position in the unit cell.
                species_expanded = []
                fractional_expanded = []
                n_H_connected_expanded = []
                xyzw = np.vstack((fractional_coords.T, np.ones((1,
                                    fractional_coords.shape[0]))))
                for am in self.affine_matrices:
                    expanded = np.array(np.dot(am, xyzw).T)
                    fractional_expanded.append(expanded[:,:3])
                    species_expanded.append(all_atoms_elements)
                    n_H_connected_expanded.append(all_atoms_n_H_connected)
                species_expanded = np.array(species_expanded).ravel()
                fractional_expanded = np.vstack(fractional_expanded)
                n_H_connected_expanded = np.hstack(n_H_connected_expanded)
            else:
                species_expanded = all_atoms_elements
                fractional_expanded = fractional_coords
                if not from_cif:
                    n_H_connected_expanded = all_atoms_n_H_connected

            with open(
                os.path.join(os.path.dirname(__file__),
                                        "atomic_scattering_params.json")) as f:
                ATOMIC_SCATTERING_PARAMS = json.load(f)
            f.close()

            zs = []
            coeffs = []
            occus = []
            dwfactors = []
            atomic_numbers = {
                1 : "H", 2 : "He", 3 : "Li", 4 : "Be", 5 : "B",
                6 : "C", 7 : "N", 8 : "O", 9 : "F", 10 : "Ne",
                11 : "Na", 12 : "Mg", 13 : "Al", 14 : "Si", 15 : "P",
                16 : "S", 17 : "Cl", 18 : "Ar", 19 : "K", 20 : "Ca",
                21 : "Sc", 22 : "Ti", 23 : "V", 24 : "Cr", 25 : "Mn",
                26 : "Fe", 27 : "Co", 28 : "Ni", 29 : "Cu",
                30 : "Zn", 31 : "Ga", 32 : "Ge", 33 : "As",
                34 : "Se", 35 : "Br", 36 : "Kr", 37 : "Rb",
                38 : "Sr", 39 : "Y", 40 : "Zr", 41 : "Nb", 42 : "Mo",
                43 : "Tc", 44 : "Ru", 45 : "Rh", 46 : "Pd",
                47 : "Ag", 48 : "Cd", 49 : "In", 50 : "Sn",
                51 : "Sb", 52 : "Te", 53 : "I", 54 : "Xe", 55 : "Cs",
                56 : "Ba", 57 : "La", 58 : "Ce", 59 : "Pr",
                60 : "Nd", 61 : "Pm", 62 : "Sm", 63 : "Eu",
                64 : "Gd", 65 : "Tb", 66 : "Dy", 67 : "Ho",
                68 : "Er", 69 : "Tm", 70 : "Yb", 71 : "Lu",
                72 : "Hf", 73 : "Ta", 74 : "W", 75 : "Re", 76 : "Os",
                77 : "Ir", 78 : "Pt", 79 : "Au", 80 : "Hg",
                81 : "Tl", 82 : "Pb", 83 : "Bi", 84 : "Po",
                85 : "At", 86 : "Rn", 87 : "Fr", 88 : "Ra",
                89 : "Ac", 90 : "Th", 91 : "Pa", 92 : "U", 93 : "Np",
                94 : "Pu", 95 : "Am", 96 : "Cm", 97 : "Bk",
                98 : "Cf", 99 : "Es", 100 : "Fm", 101 : "Md",
                102 : "No", 103 : "Lr", 104 : "Rf", 105 : "Db",
                106 : "Sg", 107 : "Bh", 108 : "Hs", 109 : "Mt",
                110 : "Ds", 111 : "Rg", 112 : "Cn", 113 : "Nh",
                114 : "Fl", 115 : "Mc", 116 : "Lv", 117 : "Ts",
                118 : "Og"
                }

            # Create a pymatgen Structure object using the dummy atom positions
            # created earlier
            dummy_structure = pmg.Structure(lattice=self.lattice,
                                species=species_expanded,
                                coords=fractional_expanded)
            i = 0
            for site in dummy_structure:
                for sp in site.species:
                    if ((not from_cif) and self.ignore_H_atoms \
                                                and self.absorb_H_Z_increase):
                        zs.append(sp.Z+n_H_connected_expanded[i])
                    else:
                        zs.append(sp.Z)
                    try:
                        if ((not from_cif)
                            and self.ignore_H_atoms
                            and self.absorb_H_Z_increase):
                            new_Z = sp.Z+n_H_connected_expanded[i]
                            c = ATOMIC_SCATTERING_PARAMS[atomic_numbers[new_Z]]
                        else:
                            c = ATOMIC_SCATTERING_PARAMS[sp.symbol]
                    except KeyError as no_key:
                        raise ValueError("Unable to calculate intensity \
                                        calculation prefix arrays"
                                        "there are no scattering coefficients\
                                        for:" " %s." % sp.symbol) from no_key
                    coeffs.append(c)
                    dwfactors.append(debye_waller_factors.get(sp.symbol, 0))
                    # DOUBLE CHECK THIS IS CORRECT FOR OCCUPANCY != 1
                    occus.append(site[site.specie.symbol])
                i += 1

            zs = np.array(zs)
            coeffs = np.array(coeffs)
            occus = np.array(occus)
            dwfactors = np.array(dwfactors)
            if from_cif:
                self.cif_dwfactors = dwfactors
            else:
                if just_asymmetric:
                    self.dwfactors_asymmetric = dwfactors
                else:
                    self.dwfactors = dwfactors

            g_hkl = np.sqrt(np.sum(np.dot(self.hkl,
                self.lattice.reciprocal_lattice_crystallographic.matrix)**2,
                axis=1))
            # Note that:
            # d_hkl = 1/g_hkl
            # tt = np.rad2deg(2 * np.arcsin(wavelength / (2*d_hkl)))
            # s = sin(theta) / wavelength = 1 / 2d = |ghkl| / 2 (d = 1/|ghkl|)
            # s = g_hkl / 2
            s2 = (g_hkl / 2)**2

            fs = []
            dw_correction = []
            for x in s2:
                fs.append(zs - 41.78214 * x * np.sum(coeffs[:, :, 0] * \
                                        np.exp(-coeffs[:, :, 1] * x), axis=1))
                dw_correction.append(np.exp(-1*dwfactors * x))
            fs = np.array(fs)
            dw_correction = np.array(dw_correction).astype(float)

            prefix = np.empty_like(fs)
            for i in range(fs.shape[0]):
                for j in range(fs.shape[1]):
                    if self.ignore_H_atoms and self.absorb_H_occu_increase:
                        new_occu = ((1 + n_H_connected_expanded[j]) / zs[j])
                        prefix[i][j] = fs[i][j] * occus[j] \
                                                * dw_correction[i][j] \
                                                * new_occu
                    else:
                        prefix[i][j] = fs[i][j] * occus[j] * dw_correction[i][j]

            return prefix

    def get_total_degrees_of_freedom(self, verbose=True):
        """
        Once all ZMs have been added to the structure object, interrogate them
        to determine the total number of degrees of freedom.
        This is automatically called by the tensor preparation code, but can
        also be called directly to print out the information.

        Args:
            verbose (bool, optional): Print out the degrees of freedom.
                Defaults to True.
        """
        total_dof = 0
        total_external = 0
        total_internal = 0
        total_position = 0
        total_rotation = 0
        for zmat in self.zmatrices:
            total_dof += zmat.degrees_of_freedom
            total_external += zmat.external_degrees_of_freedom
            total_internal += zmat.internal_degrees_of_freedom
            total_position += zmat.position_degrees_of_freedom
            total_rotation += zmat.rotation_degrees_of_freedom
        self.total_degrees_of_freedom = total_dof
        self.total_external_degrees_of_freedom = total_external
        self.total_internal_degrees_of_freedom = total_internal
        self.total_position_degrees_of_freedom = total_position
        self.total_rotation_degrees_of_freedom = total_rotation
        if verbose:
            print("Total degrees of freedom:", total_dof)
            print("Total external degrees of freedom:", total_external,
                "(pos:",total_position,"rot:",total_rotation,")")
            print("Total internal degrees of freedom:", total_internal)
        zm_torsions = []
        for zmat in self.zmatrices:
            if self.ignore_H_atoms:
                if zmat.internal_degrees_of_freedom > 0:
                    idx = zmat.torsion_refineable_indices_no_H
                    zm_torsions.append(zmat.coords_radians_no_H[:,2][idx])
            else:
                if zmat.internal_degrees_of_freedom > 0:
                    idx = zmat.torsion_refineable_indices
                    zm_torsions.append(zmat.coords_radians[:,2][idx])
        if len(zm_torsions) > 0:
            self.zm_torsions = np.hstack(zm_torsions)
        else:
            self.zm_torsions = np.array([])
        self.total_dof_calculated = True

    def add_restraint(self, zm1="", atom1=None, zm2="", atom2=None,
                        distance=None, percentage=50., indexing=1):
        """
        Add a restraint to use during the local optimisation that adds a penalty
        when atom1 and atom2 are further than "distance" angstroms apart.
        These atoms can be from the same or from different z-matrices.
        The associated ZMs can be supplied as arguments, and can either be
        integers referencing the index of the zmatrix in the Structure.zmatrices
        list, or can be the filenames of the zmatrices of interest.
        Alternatively, if all of the atoms in the structure have unique atom
        labels, then the zmatrix labels are not needed.

        e.g. atom4 in zmatrix_1.zmatrix and atom8 in zmatrix_3.zmatrix, with a
            distance of 3.0 A and percentage of 50 % can be added via:
                Structure.add_restraint(zm1="zmatrix_1.zmatrix", atom1=4,
                                        zm2="zmatrix_2.zmatrix", atom2=8,
                                        distance=3.0, percentage=50)
            Or, assuming that the zmatrices were added sequentially starting
            with zmatrix_1.zmatrix:
                Structure.add_restraint(zm1=1, atom1=4, zm2=3, atom2=8,
                                        distance=3.0, percentage=50)

            If all of the atoms have unique labels, and assuming atom1 = N1 and
            atom2 = Cl1, then the following is sufficient:
                Structure.add_restraint(atom1="N1", atom2="Cl1", distance=3.0,
                                        percentage=50)

        Users can change between 0- and 1-indexing. Default is 1-index to match
        the indexing given in the z-matrices.


        Args:
            zm1 (int or string): The index or filename of the zmatrix for atom 1
            atom1 (int): The index of atom1 in the restraint in zmatrix 1.
            zm2 (int or string): The index or filename of the zmatrix for atom 2
            atom2 (int): The index of atom1 in the restraint in zmatrix 1.
            distance (float): The distance to use for the restraint.
            percentage (float): What percentage of the minimum chi2 value to
                assign as the weight for this restraint. Defaults to 50.
            indexing (int, optional): Use python 0-indexing (i.e. first atom in
                zmatrix = 0) or normal 1-indexing (i.e. first atom in zmatrix
                = 1). Defaults to 1.

        """
        assert indexing in [0,1], "indexing must be 0 or 1"
        assert atom1 is not None and atom2 is not None, "you must specify the "\
                                                        "atoms"
        assert distance is not None, "you must specify the distance"
        check_all_zm1 = False
        check_all_zm2 = False
        if not isinstance(zm1, int):
            assert isinstance(zm1, str), "zm1 must be integer or string"
            if len(zm1) != 0:
                for i, zmat in enumerate(self.zmatrices):
                    if zm1 in zmat.filename:
                        zm1 = i + indexing
                        break
            else:
                check_all_zm1 = True
        if not isinstance(zm2, int):
            assert isinstance(zm2, str), "zm1 must be integer or string"
            if len(zm2) != 0:
                for i, zmat in enumerate(self.zmatrices):
                    if zm2 in zmat.filename:
                        zm2 = i + indexing
                        break
            else:
                check_all_zm2 = True

        assert check_all_zm1 == check_all_zm2, ("You must specify either a "
                                                "Z-matrix for both atoms or "
                                                "supply empty strings for both "
                                                "Z-matrices")

        if check_all_zm1 and check_all_zm2:
            all_atom_names = []
            all_atom_names_no_H = []
            for zmat in self.zmatrices:
                all_atom_names += zmat.atom_names
                all_atom_names_no_H += zmat.atom_names_no_H
            assert len(set(all_atom_names)) == len(all_atom_names), ("Not all "
            "atoms have unique names! Rename atom labels in Z-matrices to use "
            "atom-name-only indexing")
            atom1_idx = all_atom_names.index(atom1)
            atom2_idx = all_atom_names.index(atom2)
            atom1_no_H_idx = all_atom_names_no_H.index(atom1)
            atom2_no_H_idx = all_atom_names_no_H.index(atom2)
            restraint = [atom1_idx, atom2_idx]
            restraint_no_H = [atom1_no_H_idx, atom2_no_H_idx]
        else:
            # Now correct to get python zero-indexes if using 1-indexing
            zm1 -= indexing
            zm2 -= indexing

            if not isinstance(atom1, int):
                assert isinstance(atom1, str), "atom1 must be integer or string"
                atom1 = self.zmatrices[zm1].atom_names.index(atom1) + indexing

            if not isinstance(atom2, int):
                assert isinstance(atom2, str), "atom2 must be integer or string"
                atom2 = self.zmatrices[zm2].atom_names.index(atom2) + indexing

            atom1 -= indexing
            atom2 -= indexing

            elements = {}
            elements_no_H = {}
            n_atoms = {}
            all_elements = []
            all_elements_no_H = []
            for i, zmat in enumerate(self.zmatrices):
                elements[i] = zmat.elements
                elements_no_H[i] = zmat.elements_no_H
                n_atoms[i] = len(zmat.elements)

                all_elements += zmat.elements
                all_elements_no_H += zmat.elements_no_H

            restraint = []
            restraint_no_H = []
            for zmat, atom in zip([zm1, zm2],[atom1, atom2]):
                prev_atoms = 0
                prev_atoms_no_H = 0
                for i in range(zmat):
                    prev_atoms += len(elements[i])
                    prev_atoms_no_H += len(elements_no_H[i])

                atom_no_H = atom - elements[zmat][:atom].count("H")
                atom += prev_atoms
                atom_no_H += prev_atoms_no_H

                restraint.append(atom)
                restraint_no_H.append(atom_no_H)

        restraint += [distance, percentage]
        restraint_no_H += [distance, percentage]

        self.restraints.append(restraint)
        self.restraints_no_H.append(restraint_no_H)
