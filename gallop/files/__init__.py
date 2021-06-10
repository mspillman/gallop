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

from . import dash
from . import gsas
from . import topas

import gallop.zm_to_cart as zm_to_cart

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

    if "MD_factor" in result.keys():
        PO_axis = "# " + ",".join([str(x) for x in result["PO_axis"]])
        MD_comment = "# March-Dollase factor = " + str(
                                result["MD_factor"][chi_2 == chi_2.min()][0])
        comment = comment + "\n" + PO_axis + "\n" + MD_comment
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