# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for the web app focussed on user settings
"""
import glob
import json
import os
import torch
import streamlit as st
from collections import defaultdict

# GALLOP related
import gallop
from .. import optim


def get_options(value, options):
    if value in options:
        options.remove(value)
    else:
        st.warning("Unknown option loaded from settings file:", value)
    return [value] + options

def load_settings():
    with st.sidebar.expander(label="Load settings", expanded=False):
        with st.form(key="loadsettings"):
            filedir = os.path.dirname(gallop.__file__)
            if not os.path.exists(os.path.join(filedir,"user_settings")):
                st.error("No settings directory found")
            else:
                settings_files = list(glob.iglob(
                                os.path.join(filedir,"user_settings","*.json")))
                settings_files = [x.split(
                            "user_settings")[-1].strip("\\").strip("/")
                            for x in settings_files]
                settings_files = get_options("Default.json", settings_files)
                if len(settings_files) > 0:
                    file = st.selectbox(
                        "Choose settings file to load", settings_files,
                        key="load_settings")
                else:
                    st.error("No saved settings files found")
                st.form_submit_button(label="Load")
                filepath = os.path.join(filedir,"user_settings", file)
                with open(filepath, "r") as f:
                    json_settings = json.load(f)
                f.close()
                all_settings = {}
                all_settings.update(json_settings)
                st.success(f"Loaded settings from {file}")
    return all_settings, file

def save_settings(all_settings, filename):
    filedir = os.path.dirname(gallop.__file__)
    with st.sidebar.expander(label="Save settings", expanded=False):
        settings_name = st.text_input("Enter name to save",
                                            value=filename, max_chars=None,
                                            key=None, type='default')
        if settings_name != "":
            if st.button("Save"):
                if not os.path.exists(os.path.join(filedir,
                                                    "user_settings")):
                    os.mkdir("GALLOP_user_settings")
                if ".json" not in settings_name:
                    settings_name = settings_name + ".json"
                settings_name = os.path.join(filedir,"user_settings",
                                                    settings_name)
                if os.path.exists(settings_name):
                    os.remove(settings_name)
                with open(settings_name, "w") as f:
                    json.dump(all_settings, f, indent=4)
                f.close()
                st.success("Saved")

def general_settings(all_settings, loaded_values):
    with st.sidebar.expander(label="General", expanded=False):
        standard, optional = st.tabs(["Standard", "Optional"])
        with standard:
            all_settings["n_GALLOP_runs"] = st.number_input(
                                        "Total number of GALLOP runs",
                                        min_value=1, max_value=None,
                                        value=int(loaded_values["n_GALLOP_runs"]),
                                        step=1, format=None, key=None)
            all_settings["n_GALLOP_iters"] = st.number_input(
                                        "Total number of iterations per run",
                                        min_value=1, max_value=None,
                                        value=int(loaded_values["n_GALLOP_iters"]),
                                        step=1, format=None, key=None)
            all_settings["seed"] = int(st.number_input(
                    "Set random seed (integer >= 0), or use -1 to randomise",
                    min_value=-1, max_value=None, value=loaded_values["seed"],
                    step=1, format=None, key=None))
            if all_settings["seed"] != -1:
                st.write("Note that setting the seed does not guarantee "
                        "reproducibility due to some CUDA algorithms being "
                        "non-deterministic. See link for more information: "
                        "https://pytorch.org/docs/stable/notes/randomness.html")
                optim.seed_everything(seed=all_settings["seed"],
                                                        change_backend=False)
            all_settings["animate_structure"] = st.checkbox("Save animation of "
                                    "trajectory of best particle during LO",
                                    value=loaded_values["animate_structure"])
            if all_settings["animate_structure"]:
                st.write("Note: animation will slow down the local optimisation"
                    " and web app significantly.")
        with optional:
            structure_name = st.text_input("Structure name (optional)",
                        value=loaded_values["structure_name"], max_chars=None,
                        key=None, type='default')
            st.caption("If no name is entered, output file names will be "
                       "derived from Pawley fit filenames")
            all_settings["structure_name"] = structure_name.replace(" ", "_")
            all_settings["temperature"] = st.number_input(
                                            "Data collection temperature / K",
                                            min_value=0.0, max_value=None,
                                            value=loaded_values["temperature"],
                                            step=100.0, format=None, key=None)
            st.caption("(If > 0.0, temp will be added to CIF)")
    return all_settings

def local_settings(all_settings, loaded_values):
    # Local optimiser settings
    with st.sidebar.expander(label="Local Optimiser", expanded=False):
        standard, advanced = st.tabs(["Standard","Advanced"])
        with standard:
            options = get_options(loaded_values["optim"], ["Adam", "diffGrad"])
            all_settings["optim"] = st.selectbox("Algorithm", options)
            all_settings["n_LO_iters"] = int(st.number_input(
                                    "Number of LO steps per GALLOP iteration",
                                    min_value=1, max_value=None,
                                    value=loaded_values["n_LO_iters"],
                                    step=1, format=None, key=None))
            all_settings["ignore_H_atoms"]=st.checkbox("Remove H atoms for speed",
                                            value=loaded_values["ignore_H_atoms"],
                                            key=None)
            GPUs = []
            for i in range(torch.cuda.device_count()):
                GPUs.append(str(i)+" = "+torch.cuda.get_device_name(i))
            if len(GPUs) > 1:
                GPUs.append("Multiple GPUs")
            if torch.cuda.is_available():
                try:
                    options = get_options(loaded_values["device"],
                                                        ["Auto","CPU"]+GPUs)
                except ValueError:
                    st.error("Problem with GPU config. Setting device to CPU")
                    options = ["CPU"]
            else:
                options = ["CPU"]
            all_settings["device"] = st.selectbox("Device to perform LO",
                                    options)
            if all_settings["device"] == "Multiple GPUs":
                st.write("Note: no progress bars in multi-GPU mode")
                GPUs = st.multiselect("Select GPUs to use",GPUs[:-1],
                                                    default=GPUs[:-1])
                all_settings["particle_division"] = []
                if loaded_values["particle_division"] is None:
                    loaded_values["particle_division"] = []
                    for i in GPUs:
                        loaded_values["particle_division"].append([i.split(" =")[0],
                                                            100./float(len(GPUs))])
                for i, name in enumerate(GPUs):
                    all_settings["particle_division"].append([name.split(" = ")[0],
                        st.number_input("Enter % of particles to run on "+name,
                            min_value=0., max_value=100.,
                            value=loaded_values["particle_division"][i][1],
                            step=5.0)])
                if sum(x[1] for x in all_settings["particle_division"]) != 100.:
                    st.error("Percentages do not sum to 100 %")
            else:
                all_settings["particle_division"] = None
            all_settings["include_PO"] = st.checkbox("Include PO correction",
                                        value=loaded_values["include_PO"],
                                        key="include_PO")
            if all_settings["include_PO"]:
                if "PO_axis" not in loaded_values.keys():
                    PO_axis = "0, 0, 1"
                else:
                    PO_axis = ",".join(loaded_values["PO_axis"])
                PO_axis = st.text_input(
                            "Miller Indices for PO axis (separated by commas)",
                            value=PO_axis, max_chars=None, key=None,
                            type='default')
                try:
                    PO_axis_split = PO_axis.replace(" ","").split(",")
                    PO_axis_split = [int(x) for x in PO_axis_split]
                    all_settings["PO_axis"] = PO_axis
                except ValueError:
                    st.write("Invalid PO axis!")
        with advanced:
            find_lr = st.checkbox("Find learning rate",
                        value=loaded_values["find_lr"], key=None)
            if find_lr:
                find_lr_auto_mult = st.checkbox(
                                    "Automatically set multiplication factor",
                                    value=loaded_values["find_lr_auto_mult"],
                                    key=None)
                if not find_lr_auto_mult:
                    mult = st.number_input("Multiplication factor",
                                    min_value=0.01, max_value=None,
                                    value=loaded_values["mult"],
                                    step=0.01, format=None, key=None)
                else:
                    mult = 0.75
                lr = 0.05
            else:
                lr = st.number_input("Learning rate",
                                    min_value=0.000, max_value=None,
                                    value=loaded_values["lr"],
                                    step=0.01, format=None, key=None)
                find_lr_auto_mult = False
                mult = 1.0
            options = get_options(loaded_values["learning_rate_schedule"],
                                    ["1cycle", "sqrt", "constant"])
            learning_rate_schedule = st.selectbox("Learning rate schedule",
                                        options)
            if learning_rate_schedule == "1cycle":
                n_cooldown = int(st.number_input("Cooldown iters", min_value=1,
                                    max_value=all_settings["n_LO_iters"],
                                    value=loaded_values["n_cooldown"],
                                    step=1, format=None, key=None))
            else:
                n_cooldown = loaded_values["n_cooldown"]

            options = get_options(loaded_values["loss"],
                                ["sum", "xlogx", "sse"])
            loss = st.selectbox("Loss function to minimise",
                                options)
            reflection_percentage = st.number_input(
                                "% of reflections to include",
                                min_value=0.1, max_value=100.,
                                value=loaded_values["reflection_percentage"],
                                step=10., format=None, key=None)

            percentage_cutoff = st.number_input(
                                    "% correlation to ignore",
                                    min_value=0.0, max_value=100.,
                                    value=loaded_values["percentage_cutoff"],
                                    step=20., format=None, key=None)
            include_dw_factors = st.checkbox("Include DW factors in chi2 calcs",
                                    value=loaded_values["include_dw_factors"],
                                    key=None)
            memory_opt = st.checkbox(
                            "Reduce local opt speed to improve GPU memory use",
                            value=loaded_values["memory_opt"], key=None)
            torsion_shadowing = st.checkbox("Use torsion shadowing",
                                    value=loaded_values["torsion_shadowing"])
            if torsion_shadowing:
                Z_prime = int(st.number_input("Enter Z' of structure",
                                min_value=1, max_value=None,
                                value=int(loaded_values["Z_prime"]), step=1,
                                format=None, key=None))
                shadow_iters = int(st.number_input("Number of shadowing iterations",
                                min_value=0,
                                max_value=int(all_settings["n_GALLOP_iters"]),
                                value=int(loaded_values["shadow_iters"]),
                                step=1, format=None, key=None))
            else:
                Z_prime = False
                shadow_iters = 0
            use_distance_restraints = st.checkbox("Use distance restraints",
                                    value=loaded_values["use_distance_restraints"])
            if use_distance_restraints:
                n_distance_restraints = int(st.number_input("Enter number of distance restraints to use",
                                        min_value=0, max_value=None,
                                        value=loaded_values["n_distance_restraints"],
                                        step=1, format=None, key=None))
                distance_restraints = loaded_values["distance_restraints"]
                st.write("Enter the atom labels, distance and \
                        weight (separated by commas, e.g. C1,C2,1.54,0.5)")
                if distance_restraints is None:
                    distance_restraints = defaultdict(str)
                for i in range(n_distance_restraints):
                    #st.write()
                    r = st.text_input(f"Distance restraint {i+1}:",key=f"dr_{i+1}",
                                        value=distance_restraints[str(i)])
                    distance_restraints[str(i)] = r
            else:
                distance_restraints = None
                n_distance_restraints = 0

            use_angle_restraints = st.checkbox("Use angle restraints",
                                    value=loaded_values["use_angle_restraints"])
            if use_angle_restraints:
                n_angle_restraints = int(st.number_input("Enter number of angle restraints to use",
                                        min_value=0, max_value=None,
                                        value=loaded_values["n_angle_restraints"],
                                        step=1, format=None, key=None))
                angle_restraints = loaded_values["angle_restraints"]
                st.write("Enter the atom labels (3 or 4), angle and \
                        weight (separated by commas, e.g. C1,C2,C3,121.1,0.5 \
                        or C1,C2,C3,C4,121.1,0.5)")
                if angle_restraints is None:
                    angle_restraints = defaultdict(str)
                for i in range(n_angle_restraints):
                    #st.write()
                    r = st.text_input(f"Angle restraint {i+1}:",key=f"ar_{i+1}",
                                        value=angle_restraints[str(i)])
                    angle_restraints[str(i)] = r
            else:
                angle_restraints = None
                n_angle_restraints = 0

            use_torsion_restraints = st.checkbox("Use torsion restraints",
                                    value=loaded_values["use_torsion_restraints"])
            if use_torsion_restraints:
                n_torsion_restraints = int(st.number_input("Enter number of torsion restraints to use",
                                        min_value=0, max_value=None,
                                        value=loaded_values["n_torsion_restraints"],
                                        step=1, format=None, key=None))
                torsion_restraints = loaded_values["torsion_restraints"]
                st.write("Enter the atom labels, torsion angle and \
                        weight (separated by commas, e.g. C1,C2,C3,C4,121.1,0.5)")
                if torsion_restraints is None:
                    torsion_restraints = defaultdict(str)
                for i in range(n_torsion_restraints):
                    #st.write()
                    r = st.text_input(f"Torsion restraint {i+1}:",key=f"tr_{i+1}",
                                        value=torsion_restraints[str(i)])
                    torsion_restraints[str(i)] = r
            else:
                torsion_restraints = None
                n_torsion_restraints = 0
            options = get_options(loaded_values["restraint_weight_type"],
                            ["min_chi2", "chi2", "constant"])
            restraint_weight_type = st.selectbox("Restraint weight type", options)

    all_settings["find_lr"] = find_lr
    all_settings["find_lr_auto_mult"] = find_lr_auto_mult
    all_settings["mult"] = mult
    all_settings["lr"] = lr
    all_settings["learning_rate_schedule"] = learning_rate_schedule
    all_settings["n_cooldown"] = n_cooldown
    all_settings["loss"] = loss
    all_settings["reflection_percentage"] = reflection_percentage
    all_settings["include_dw_factors"] = include_dw_factors
    all_settings["memory_opt"] = memory_opt
    all_settings["percentage_cutoff"] = percentage_cutoff
    all_settings["torsion_shadowing"] = torsion_shadowing
    all_settings["Z_prime"] = Z_prime
    all_settings["shadow_iters"] = shadow_iters
    all_settings["use_distance_restraints"] = use_distance_restraints
    all_settings["n_distance_restraints"] = n_distance_restraints
    all_settings["distance_restraints"] = distance_restraints
    all_settings["use_angle_restraints"] = use_angle_restraints
    all_settings["n_angle_restraints"] = n_angle_restraints
    all_settings["angle_restraints"] = angle_restraints
    all_settings["use_torsion_restraints"] = use_torsion_restraints
    all_settings["n_torsion_restraints"] = n_torsion_restraints
    all_settings["torsion_restraints"] = torsion_restraints
    all_settings["restraint_weight_type"] = restraint_weight_type
    return all_settings

def pso_settings(all_settings, loaded_values):
    # Particle Swarm settings
    with st.sidebar.expander(label="Particle Swarm", expanded=False):
        standard, advanced = st.tabs(["Standard", "Advanced"])
        with standard:
            all_settings["n_swarms"] = int(st.number_input("Number of swarms",
                                    min_value=1, max_value=None,
                                    value=loaded_values["n_swarms"], step=1,
                                    format=None, key=None))
            all_settings["swarm_size"] = int(st.number_input(
                                        "Number of particles per swarm",
                                        min_value=1,max_value=None,
                                        value=loaded_values["swarm_size"],
                                        step=100, format=None, key=None))
            st.write("Total particles =",
                                all_settings["n_swarms"]*all_settings["swarm_size"])
            all_settings["global_update"] = st.checkbox(
                                            "Periodic all-swarm global update",
                                            value=loaded_values["global_update"],
                                            key=None)
            if all_settings["global_update"]:
                all_settings["global_update_freq"] = int(st.number_input(
                            "All-swarm global update frequency",
                            min_value=1, max_value=int(all_settings["n_GALLOP_iters"]),
                            value=int(loaded_values["global_update_freq"]),
                            step=1, format=None, key=None))
            else:
                all_settings["global_update_freq"]=int(all_settings["n_GALLOP_iters"])
            all_settings["randomise_worst"] = st.checkbox(
                            "Periodically randomise the worst performing particles",
                            value=loaded_values["randomise_worst"], key=None)
            if all_settings["randomise_worst"]:
                all_settings["randomise_percentage"] = st.number_input(
                            "Percentage of worst performing particles to randomise",
                            min_value=0.0, max_value=100.0,
                            value=float(loaded_values["randomise_percentage"]),
                            step=10.0, format=None, key=None)
                all_settings["randomise_freq"] = int(st.number_input(
                            "Randomisation frequency",
                            min_value=1, max_value=int(all_settings["n_GALLOP_iters"]),
                            value=int(loaded_values["randomise_freq"]),
                            step=1, format=None, key=None))

            else:
                all_settings["randomise_percentage"] = 0
                all_settings["randomise_freq"] = all_settings["n_GALLOP_iters"]+1
        with advanced:
            c1 = st.number_input("c1 (social component)", min_value=0.0,
                                    max_value=None, value=loaded_values["c1"],
                                    step=0.1)
            c2 = st.number_input("c2 (cognitive component)", min_value=0.0,
                                    max_value=None, value=loaded_values["c2"],
                                    step=0.1)
            options = get_options(loaded_values["inertia_type"],
                            ["ranked", "random", "constant"])
            inertia_type = st.selectbox("Inertia type", options)
            if inertia_type == "constant":
                inertia = st.number_input("Inertia", min_value=0.0,
                                    max_value=None,
                                    value=loaded_values["inertia"], step=0.01,
                                    format=None, key=None)
                inertia_bounds = (0.4,0.9)
            else:
                lower = st.number_input("Lower inertia bound", min_value=0.,
                                    max_value=None,
                                    value=loaded_values["inertia_bounds"][0],
                                    step=0.05)
                upper = st.number_input("Upper inertia bound", min_value=lower,
                                    max_value=None,
                                    value=loaded_values["inertia_bounds"][1],
                                    step=0.05)
                inertia_bounds = (lower, upper)
                inertia = 0.7
            limit_velocity = st.checkbox("Limit velocity",
                                value=loaded_values["limit_velocity"], key=None)
            if limit_velocity:
                vmax = st.number_input("Maximum absolute velocity",
                                    min_value=0.,max_value=None,
                                    value=loaded_values["vmax"],
                                    step=0.05)
            else:
                vmax = 1.0
    all_settings["c1"] = c1
    all_settings["c2"] = c2
    all_settings["inertia_type"] = inertia_type
    all_settings["inertia"] = inertia
    all_settings["inertia_bounds"] = inertia_bounds
    all_settings["limit_velocity"] = limit_velocity
    all_settings["vmax"] = vmax
    return all_settings

def get_all_settings(loaded_values):
    # Get GALLOP settings. Menu appears on the sidebar, with expandable sections
    # for general, local optimiser and particle swarm optimiser respectively.
    all_settings = {}
    all_settings = general_settings(all_settings, loaded_values)

    all_settings = local_settings(all_settings, loaded_values)

    all_settings = pso_settings(all_settings, loaded_values)

    return all_settings