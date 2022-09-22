# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for the web app for working with gallop structures
"""
import os
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt

from .. import z_matrix
from .. import optim



def get_name(all_settings, pawley_program, sdi, gpx, out, cif, ins):
    name = all_settings["structure_name"]
    if name == "Enter_structure_name" or len(name) == 0:
        if pawley_program == "DASH":
            structure_name = os.path.split(sdi)[-1].split(".sdi")[0]
        elif pawley_program == "GSAS-II":
            structure_name = os.path.split(gpx)[-1].split(".gpx")[0]
        elif pawley_program == "TOPAS (experimental)":
            structure_name = os.path.split(out)[-1].split(".out")[0]
        else:
            assert (cif is not None or ins is not None), "You must supply \
                            cell and space group info via a cif or ins file"
            if cif is not None:
                structure_name = os.path.split(cif)[-1].split(".cif")[0]
            else:
                structure_name = os.path.split(ins)[-1].split(".ins")[0]
        name = structure_name
    return name

def add_data(struct, all_settings, pawley_program, sdi, gpx, out, cif, ins, hkl):
    if pawley_program == "DASH":
        struct.add_data(sdi, source="DASH",
                    percentage_cutoff=all_settings["percentage_cutoff"])
    elif pawley_program == "GSAS-II":
        struct.add_data(gpx, source="GSAS",
                    percentage_cutoff=all_settings["percentage_cutoff"])
    elif pawley_program == "TOPAS (experimental)":
        struct.add_data(out, source="TOPAS",
                    percentage_cutoff=all_settings["percentage_cutoff"])
    else:
        if cif is not None:
            struct.add_data(cif, hklfile=hkl, source="SHELX")
        else:
            struct.add_data(ins, hklfile=hkl, source="SHELX")

def add_zmatrices(struct, zms, all_settings):
    for z in sorted(zms):
        check = z_matrix.Z_matrix(z)
        if all_settings["ignore_H_atoms"] and not check.H_atom_torsion_defs:
            struct.add_zmatrix(z)
        elif all_settings["ignore_H_atoms"] and check.H_atom_torsion_defs:
            st.markdown("**Problem with z-matrix "+z+" - H-atoms used to "
                "define refineable torsion angles. Please generate a new ZM "
                "to allow H-atoms to be ignored or refresh the page and "
                "uncheck the ignore H atoms box.**")
            st.write("Attempting to continue with H-atoms included")
            struct.ignore_H_atoms = False
            struct.add_zmatrix(z)
        else:
            struct.add_zmatrix(z)

def get_minimiser_settings(struct, all_settings, use_profile, step):
    minimiser_settings = optim.local.get_minimiser_settings(struct)
    minimiser_settings["streamlit"] = True
    minimiser_settings["include_dw_factors"] = \
                                        all_settings["include_dw_factors"]
    minimiser_settings["n_iterations"] = all_settings["n_LO_iters"]
    minimiser_settings["learning_rate_schedule"] = \
                                    all_settings["learning_rate_schedule"]
    n_refs = int(np.ceil((all_settings["reflection_percentage"]/100.)
                *len(struct.hkl)))
    if n_refs > len(struct.hkl):
        n_refs = len(struct.hkl)
    minimiser_settings["n_reflections"] = n_refs

    if all_settings["learning_rate_schedule"] == "1cycle":
        minimiser_settings["n_cooldown"] = all_settings["n_cooldown"]

    if all_settings["device"] == "CPU":
        minimiser_settings["device"] = torch.device("cpu")
    elif all_settings["device"] == "Auto":
        minimiser_settings["device"] = None
    elif all_settings["device"] == "Multiple GPUs":
        minimiser_settings["device"] = torch.device("cuda:" +
                            all_settings["particle_division"][0][0])
    else:
        minimiser_settings["device"] = torch.device("cuda:"+
                                    all_settings["device"].split(" = ")[0])
    minimiser_settings["loss"] = all_settings["loss"]
    minimiser_settings["optimizer"] = all_settings["optim"].lower()
    minimiser_settings["torsion_shadowing"] = all_settings["torsion_shadowing"]
    minimiser_settings["Z_prime"] = all_settings["Z_prime"]
    if all_settings["include_PO"]:
        minimiser_settings["include_PO"] = True
        PO_axis_split = all_settings["PO_axis"].replace(" ","").split(",")
        PO_axis_split = [int(x) for x in PO_axis_split]
        minimiser_settings["PO_axis"] = PO_axis_split

    if all_settings["use_distance_restraints"]:
        if all_settings["distance_restraints"] is not None:
            for r in all_settings["distance_restraints"].keys():
                r = all_settings["distance_restraints"][r]
                r = r.replace(" ","").split(",")
                struct.add_restraint({"type" : "distance",
                                        "atom1" : r[0],
                                        "atom2" : r[1],
                                        "value" : float(r[2]),
                                        "weight" : float(r[3])})
        minimiser_settings["use_restraints"] = True

    if all_settings["use_angle_restraints"]:
        if all_settings["angle_restraints"] is not None:
            for r in all_settings["angle_restraints"].keys():
                r = all_settings["angle_restraints"][r]
                r = r.replace(" ","").split(",")
                if len(r) == 5:
                    struct.add_restraint({"type" : "angle",
                                        "atom1" : r[1],
                                        "atom2" : r[0],
                                        "atom3" : r[1],
                                        "atom4" : r[2],
                                        "value" : float(r[3]),
                                        "weight" : float(r[4])})
                else:
                    struct.add_restraint({"type" : "angle",
                                        "atom1" : r[0],
                                        "atom2" : r[1],
                                        "atom3" : r[2],
                                        "atom4" : r[3],
                                        "value" : float(r[4]),
                                        "weight" : float(r[5])})
        minimiser_settings["use_restraints"] = True

    if all_settings["use_torsion_restraints"]:
        if all_settings["torsion_restraints"] is not None:
            for r in all_settings["torsion_restraints"].keys():
                r = all_settings["torsion_restraints"][r]
                r = r.replace(" ","").split(",")
                struct.add_restraint({"type" : "torsion",
                                        "atom1" : r[0],
                                        "atom2" : r[1],
                                        "atom3" : r[2],
                                        "atom4" : r[3],
                                        "value" : float(r[4]),
                                        "weight" : float(r[5])})
        minimiser_settings["use_restraints"] = True

    minimiser_settings["restraint_weight_type"] = all_settings["restraint_weight_type"]
    minimiser_settings["profile"] = use_profile
    minimiser_settings["step"] = step

    if all_settings["animate_structure"]:
        minimiser_settings["save_trajectories"] = True
    return minimiser_settings

def get_swarm(struct, all_settings):
    n_particles = all_settings["swarm_size"]*all_settings["n_swarms"]
    if all_settings["inertia_type"] == "constant":
        inertia = all_settings["inertia"]
    else:
        inertia = all_settings["inertia_type"]
    swarm = optim.swarm.Swarm(
                Structure = struct,
                n_particles=int(n_particles),
                n_swarms = int(all_settings["n_swarms"]),
                global_update = all_settings["global_update"],
                global_update_freq = all_settings["global_update_freq"],
                inertia_bounds = all_settings["inertia_bounds"],
                inertia = inertia,
                c1 = all_settings["c1"],
                c2 = all_settings["c2"],
                limit_velocity = all_settings["limit_velocity"],
                vmax = all_settings["vmax"])
    return swarm

def find_learning_rate(all_settings, minimiser_settings, struct,
                        external, internal):
    # Learning rate finder interface
    col1, col2 = st.columns(2)
    with col1:
        st.write("Finding the learning rate")
    if not all_settings["find_lr_auto_mult"]:
        st.write("Using multiplication_factor", all_settings["mult"])
    try:
        if all_settings["device"] == "Multiple GPUs":
            percentage = all_settings["particle_division"][0][1] / 100.
            end = int(np.ceil(len(external)*percentage))
        else:
            end = len(external)
        lr = optim.local.find_learning_rate(struct, external=external[0:end],
                                internal=internal[0:end],
                                multiplication_factor=all_settings["mult"],
                                min_lr=-4, max_lr=np.log10(0.15),
                                plot=False,
                                logplot=False,
                                minimiser_settings = minimiser_settings)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        lrs = lr[0].copy()
        losses = lr[1].copy()
        lrs -= lrs.min()
        losses -= losses.min()
        lrs /= lrs.max()
        losses /= losses.max()
        with col2:
            with st.expander(label=" ", expanded=False):
                minpoint = np.argmin(losses)
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 2))
                ax[0].plot(lr[0], lr[1])
                ax[1].plot(lrs[minpoint:]-lrs[minpoint:].min(),
                    lrs[minpoint:]-lrs[minpoint:].min(),":",alpha=0.5,c="k")
                ax[1].plot(lrs[minpoint:]-lrs[minpoint:].min(),
                    0.5*(lrs[minpoint:]-lrs[minpoint:].min()),"-.",
                    alpha=0.5,c="k")
                ax[1].plot(lrs[minpoint:]-lrs[minpoint:].min(),
                    0.25*(lrs[minpoint:]-lrs[minpoint:].min()),"--",
                    alpha=0.5,c="k")
                ax[1].plot(lrs[minpoint:]-lrs[minpoint:].min(),
                            losses[minpoint:])
                gradient = ((losses[-1] - losses[minpoint])
                            / (lrs[-1] - lrs[minpoint]))
                ax[1].plot(lrs[minpoint:]-lrs[minpoint:].min(),
                            gradient*(lrs[minpoint:]-lrs[minpoint:].min()),
                            c="r")
                ax[0].set_xlabel('learning rate')
                ax[1].set_xlabel('normalised learning rate')
                ax[0].set_ylabel('sum of $\\chi^{2}$ vals')
                ax[1].set_ylabel('rescaled sum')
                ax[1].legend(["y=x","y=0.5x","y=0.25x","rescaled sum", "approx"],
                                loc=2, prop={'size': 8})
                st.pyplot(fig)
                if all_settings["find_lr_auto_mult"]:
                    final_1 = (lrs[minpoint:]-lrs[minpoint:].min())[-1]
                    final_0pt5 = 0.5 * final_1
                    final_0pt25 = 0.25 * final_1
                    if losses[-1] < final_0pt25:
                        mult = 1.0
                    elif losses[-1] < final_0pt5:
                        mult = 0.75
                    else:
                        mult = 0.5
                    #mult = max(0.5, min(1.0, 1.25
                    #                - (losses[-1] / (lrs[-1] - lrs[minpoint]))))
                    minimiser_settings["learning_rate"] = lr[-1] * mult
                    st.write("Learning rate:", np.around(lr[-1] * mult,4))
                    st.write("Mult:", np.around(mult, 4))
                    all_settings["mult"] = mult
                else:
                    minimiser_settings["learning_rate"] = lr[-1]
                    st.write("Learning rate:", np.around(lr[-1], 4))
        failed = False
    except RuntimeError as e:
        if "memory" in str(e):
            st.error("GPU memory error! Reset GALLOP, then:")
            st.write("Try reducing total number of particles, the number "
            "of reflections or use the Reduce performance option "
            "in Local optimiser settings.")
            st.write("Error code below:")
            st.write("")
            st.text(str(e))
        else:
            st.error("An unknown error occured")
            st.write("")
            st.text(str(e))
        failed = True
    all_settings["lr"] = minimiser_settings["learning_rate"]

    return all_settings, minimiser_settings, failed