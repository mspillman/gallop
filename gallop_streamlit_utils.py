# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for the web app
"""

# Others needed
import glob
import base64
import json
import datetime
import os
from io import StringIO
import torch
import py3Dmol
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# GALLOP related
from gallop import optimiser
from gallop import tensor_prep
from gallop import chi2
from gallop import files

settings = ["structure_name", "n_GALLOP_iters", "seed",
            "optim", "n_LO_iters", "ignore_H_atoms", "device",
            "particle_division","include_PO","PO_axis","find_lr",
            "find_lr_auto_mult", "mult", "learning_rate_schedule",
            "n_cooldown", "loss", "reflection_percentage", "include_dw_factors",
            "memory_opt", "percentage_cutoff", "torsion_shadowing",
            "Z_prime", "shadow_iters", "n_swarms", "swarm_size",
            "global_update","global_update_freq", "randomise_worst",
            "randomise_percentage", "randomise_freq", "c1", "c2",
            "inertia","inertia_bounds", "limit_velocity", "vmax",
            "lr"]

def get_options(value, options):
    options.remove(value)
    return [value] + options

def improve_GPU_memory_use(struct, minimiser_settings):
    """
    This function improves the use of GPU memory, at the expense of worse
    performance in terms of number of chi2 evaluations per second.
    This works by initialising the JIT compilation to optimise for gradient-free
    chi2 calcs, which slows down the backward pass used to find the gradients,
    but reduces the memory requirements a little. Doesn't result in massive
    benefits so not really recommended unless you really need an extra few
    particles.
    """
    swarm = optimiser.Swarm(Structure=struct, n_particles=1, n_swarms=1,
                            global_update_freq=20)
    external, internal = swarm.get_initial_positions()
    tensors = tensor_prep.get_all_required_tensors(
            struct, external=external, internal=internal,
            n_samples=external.shape[0], device=minimiser_settings["device"],
            dtype=minimiser_settings["dtype"],
            n_reflections=minimiser_settings["n_reflections"],
            verbose=minimiser_settings["verbose"],
            include_dw_factors=minimiser_settings["include_dw_factors"],
            requires_grad=False)
    chi_2 = chi2.get_chi_2(**tensors)
    del tensors, chi_2


def load_settings():
    with st.sidebar.beta_expander(label="Load settings", expanded=False):
        if not os.path.exists("GALLOP_user_settings"):
            st.error("No settings directory found")
        else:
            settings_files = list(glob.iglob(
                            os.path.join("GALLOP_user_settings","*.json")))
            settings_files = [x.replace(
                        "GALLOP_user_settings","").strip("\\").strip("/")
                        for x in settings_files]
            settings_files = get_options("Default.json", settings_files)
            if len(settings_files) > 0:
                file = st.selectbox(
                    "Choose settings file to load", settings_files, key="load_settings")
            else:
                st.error("No saved settings files found")
            filepath = os.path.join("GALLOP_user_settings", file)
            with open(filepath, "r") as f:
                json_settings = json.load(f)
            f.close()
            all_settings = {}
            all_settings.update(json_settings)
            st.success(f"Loaded settings from {file}")
    return all_settings, file

def save_settings(all_settings, filename):
    with st.sidebar.beta_expander(label="Save settings", expanded=False):
        settings_name = st.text_input("Enter name to save",
                                            value=filename, max_chars=None,
                                            key=None, type='default')
        if settings_name != "":
            if st.button("Save"):
                if not os.path.exists("GALLOP_user_settings"):
                    os.mkdir("GALLOP_user_settings")
                if ".json" not in settings_name:
                    settings_name = settings_name + ".json"
                settings_name = os.path.join("GALLOP_user_settings",
                                                    settings_name)
                if os.path.exists(settings_name):
                    os.remove(settings_name)
                with open(settings_name, "w") as f:
                    json.dump(all_settings, f, indent=4)
                f.close()
                st.success("Saved")

def get_all_settings(loaded_values):
    # Get GALLOP settings. Menu appears on the sidebar, with expandable sections
    # for general, local optimiser and particle swarm optimiser respectively.
    all_settings = {}
    with st.sidebar.beta_expander(label="General", expanded=False):
        structure_name = st.text_input("Structure name (optional)",
                        value=loaded_values["structure_name"], max_chars=None,
                        key=None, type='default')
        all_settings["structure_name"] = structure_name.replace(" ", "_")
        all_settings["n_GALLOP_iters"] = st.number_input(
                                        "Total number of GALLOP iterations",
                                        min_value=1, max_value=None,
                                        value=loaded_values["n_GALLOP_iters"],
                                        step=1, format=None, key=None)

        all_settings["seed"] = st.number_input(
                    "Set random seed (integer >= 0), or use -1 to randomise",
                    min_value=-1, max_value=None, value=loaded_values["seed"],
                    step=1, format=None, key=None)
        if all_settings["seed"] != -1:
            st.write("Note that setting the seed does not guarantee "
                    "reproducibility due to some CUDA algorithms being "
                    "non-deterministic. See link for more information: "
                    "https://pytorch.org/docs/stable/notes/randomness.html")
            optimiser.seed_everything(seed=all_settings["seed"],
                                                        change_backend=False)
        all_settings["animate_structure"] = st.checkbox("Save animation of "
                                "trajectory of best particle during LO",
                                value=loaded_values["animate_structure"])
        if all_settings["animate_structure"]:
            st.write("Note: animation will slow down the local optimisation.")

    # Local optimiser settings
    with st.sidebar.beta_expander(label="Local Optimiser", expanded=False):
        options = get_options(loaded_values["optim"], ["Adam", "diffGrad"])
        all_settings["optim"] = st.selectbox("Algorithm", options)
        all_settings["n_LO_iters"] = st.number_input(
                                "Number of LO steps per GALLOP iteration",
                                min_value=1, max_value=None,
                                value=loaded_values["n_LO_iters"],
                                step=1, format=None, key=None)
        all_settings["ignore_H_atoms"]=st.checkbox("Remove H atoms for speed",
                                        value=loaded_values["ignore_H_atoms"],
                                        key=None)
        GPUs = []
        for i in range(torch.cuda.device_count()):
            GPUs.append(str(i)+" = "+torch.cuda.get_device_name(i))
        if len(GPUs) > 1:
            GPUs.append("Multiple GPUs")
        options = get_options(loaded_values["device"], ["Auto","CPU"]+GPUs)
        all_settings["device"] = st.selectbox("Device to perform LO",
                                options)
        if all_settings["device"] == "Multiple GPUs":
            GPUs = st.multiselect("Select GPUs to use",GPUs[:-1],
                                                default=GPUs[:-1])
            all_settings["particle_division"] = []
            for i in GPUs:
                all_settings["particle_division"].append([i.split(" = ")[0],
                    st.number_input("Enter % of particles to run on "+i,
                        min_value=0., max_value=100.,
                        value=loaded_values["particle_division"][i][1])])
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
                PO_axis = PO_axis.replace(" ","").split(",")
                PO_axis = [int(x) for x in PO_axis]
                all_settings["PO_axis"] = PO_axis
            except ValueError:
                st.write("Invalid PO axis!")
        st.markdown("***Advanced***")
        show_advanced_lo = st.checkbox("Show advanced options", value=False,
                                        key="show_advanced_lo")
        if show_advanced_lo:
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
                n_cooldown = st.number_input("Cooldown iters", min_value=1,
                                    max_value=all_settings["n_LO_iters"],
                                    value=loaded_values["n_cooldown"],
                                    step=1, format=None, key=None)
            else:
                n_cooldown = loaded_values["n_cooldown"]
            torsion_shadowing = st.checkbox("Use torsion shadowing",
                                    value=loaded_values["torsion_shadowing"])
            if torsion_shadowing:
                Z_prime = st.number_input("Enter Z' of structure",
                                min_value=1, max_value=None,
                                value=loaded_values["Z_prime"], step=1,
                                format=None, key=None)
                shadow_iters = st.number_input("Number of shadowing iterations",
                                min_value=0,
                                max_value=all_settings["n_GALLOP_iters"],
                                value=loaded_values["shadow_iters"],
                                step=1, format=None, key=None)
            else:
                Z_prime = False
                shadow_iters = 0
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
            use_restraints = st.checkbox("Use distance restraints",
                                        value=loaded_values["use_restraints"])
            if use_restraints:
                n_restraints = st.number_input("Enter number of restraints to use",
                                        min_value=0, max_value=None,
                                        value=loaded_values["n_restraints"],
                                        step=1, format=None, key=None)
                restraints = loaded_values["restraints"]
                if restraints is None:
                    restraints = defaultdict(str)
                for i in range(n_restraints):
                    st.write(f"Restraint {i+1}:")
                    r = st.text_input("Enter the atom labels, distance and \
                        % weight (separated by commas, e.g. C1,C2,1.54,50)",
                        key=f"r_{i+1}", value=restraints[str(i)])
                    restraints[str(i)] = r
            else:
                restraints = None
                n_restraints = 0
        else:
            find_lr = loaded_values["find_lr"]
            find_lr_auto_mult = loaded_values["find_lr_auto_mult"]
            lr = loaded_values["lr"]
            mult = loaded_values["mult"]
            learning_rate_schedule = loaded_values["learning_rate_schedule"]
            n_cooldown = loaded_values["n_cooldown"]
            loss = loaded_values["loss"]
            reflection_percentage = loaded_values["reflection_percentage"]
            include_dw_factors = loaded_values["include_dw_factors"]
            memory_opt = loaded_values["memory_opt"]
            percentage_cutoff = loaded_values["percentage_cutoff"]
            torsion_shadowing = loaded_values["torsion_shadowing"]
            Z_prime = loaded_values["Z_prime"]
            shadow_iters = loaded_values["shadow_iters"]
            use_restraints = loaded_values["use_restraints"]
            n_restraints = loaded_values["n_restraints"]
            restraints = loaded_values["restraints"]

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
    all_settings["use_restraints"] = use_restraints
    all_settings["n_restraints"] = n_restraints
    all_settings["restraints"] = restraints

    # Particle Swarm settings
    with st.sidebar.beta_expander(label="Particle Swarm", expanded=False):
        all_settings["n_swarms"] = st.number_input("Number of swarms",
                                min_value=1, max_value=None,
                                value=loaded_values["n_swarms"], step=1,
                                format=None, key=None)
        all_settings["swarm_size"] = st.number_input(
                                    "Number of particles per swarm",
                                    min_value=1,max_value=None, 
                                    value=loaded_values["swarm_size"],
                                    step=100, format=None, key=None)
        st.write("Total particles =",
                            all_settings["n_swarms"]*all_settings["swarm_size"])
        all_settings["global_update"] = st.checkbox(
                                        "Periodic all-swarm global update",
                                        value=loaded_values["global_update"],
                                        key=None)
        if all_settings["global_update"]:
            all_settings["global_update_freq"] = st.number_input(
                        "All-swarm global update frequency",
                        min_value=1, max_value=all_settings["n_GALLOP_iters"],
                        value=loaded_values["global_update_freq"],
                        step=1, format=None, key=None)
        else:
            all_settings["global_update_freq"]=all_settings["n_GALLOP_iters"]
        all_settings["randomise_worst"] = st.checkbox(
                        "Periodically randomise the worst performing particles",
                        value=loaded_values["randomise_worst"], key=None)
        if all_settings["randomise_worst"]:
            all_settings["randomise_percentage"] = st.number_input(
                        "Percentage of worst performing particles to randomise",
                        min_value=0.0, max_value=100.0,
                        value=loaded_values["randomise_percentage"],
                        step=10.0, format=None, key=None)
            all_settings["randomise_freq"] = st.number_input(
                        "Randomisation frequency",
                        min_value=1, max_value=all_settings["n_GALLOP_iters"],
                        value=loaded_values["randomise_freq"],
                        step=1, format=None, key=None)
        else:
            all_settings["randomise_percentage"] = 0
            all_settings["randomise_freq"] = all_settings["n_GALLOP_iters"]+1
        st.markdown("***Advanced***")
        show_advanced_pso = st.checkbox("Show advanced options", value=False,
                                        key="show_advanced_pso")
        if show_advanced_pso:
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
        else:
            c1 = loaded_values["c1"]
            c2 = loaded_values["c2"]
            inertia_type = loaded_values["inertia_type"]
            inertia = loaded_values["inertia"]
            inertia_bounds = loaded_values["inertia_bounds"]
            limit_velocity = loaded_values["limit_velocity"]
            vmax = loaded_values["vmax"]
    all_settings["c1"] = c1
    all_settings["c2"] = c2
    all_settings["inertia_type"] = inertia_type
    all_settings["inertia"] = inertia
    all_settings["inertia_bounds"] = inertia_bounds
    all_settings["limit_velocity"] = limit_velocity
    all_settings["vmax"] = vmax

    return all_settings


def sidebar():
    all_settings, filename = load_settings()

    all_settings = get_all_settings(all_settings)

    #all_settings = load_save_settings(all_settings)

    save_settings(all_settings, filename)

    st.sidebar.write("")
    reset = st.sidebar.button("Reset GALLOP")
    # This code for resetting streamlit is a bit of an ugly hack, but it
    # works. If there's a more elegant way to do it, it's not easy to find!
    # The code writes out a short script - batch on Windows, otherwise a .py
    # which is then called. This kills the running streamlit process, and then
    # starts a new one.
    if reset:
        # Windows
        if os.name == "nt":
            lines = ["taskkill /IM \"streamlit.exe\" /F\n",
                    "streamlit run .\\gallop_streamlit.py"]
            with open("reset.bat", "w") as reset_script:
                reset_script.writelines(lines)
            reset_script.close()
            os.system("START /B reset.bat")
        # Linux
        else:
            lines = ["import os\n",
                "import time\n",
                "os.system(\"pkill -f streamlit &> /dev/null&\")\n",
                "time.sleep(1)\n",
                "os.system(\"streamlit run gallop_streamlit.py &> /dev/null\")"]
            with open("reset.py", "w") as reset_script:
                reset_script.writelines(lines)
            reset_script.close()
            os.system("python3 reset.py &> /dev/null&")

    return all_settings






def get_files():
    # Upload files using uploader or select from examples
    col1, col2 = st.beta_columns(2)
    with col1:
        file_source = st.radio("Upload files or select from examples",
                                        ["Upload files","Use examples"])
    if file_source == "Use examples":
        pawley_program = "DASH"
        example = st.selectbox("Select example structure",
                    ["Verapamil hydrochloride", "Famotidine form B"])
        uploaded_files = []
        clear_files = False
        load_settings_from_file = False
    else:
        with col2:
            pawley_program = st.radio("Choose Pawley refinement program",
                                    ["DASH","GSAS-II", "TOPAS"])
        upload_key = 1
        if pawley_program == "DASH":
            uploaded_files = st.file_uploader("Upload DASH Pawley files and \
                                    Z-matrices",
                                    accept_multiple_files=True,
                                    type=["zmatrix", "sdi", "hcv", "tic", "dsl",
                                        "dbf", "json"],
                                    key=upload_key)
        elif pawley_program == "GSAS-II":
            uploaded_files = st.file_uploader("Upload GSAS-II project file and\
                                    Z-matrices",
                                    accept_multiple_files=True,
                                    type=["zmatrix", "gpx", "json"],
                                    key=upload_key)
        else:
            uploaded_files = st.file_uploader("Upload TOPAS .out file and\
                                    Z-matrices",
                                    accept_multiple_files=True,
                                    type=["zmatrix", "out", "json"],
                                    key=upload_key)
        col1, col2 = st.beta_columns([2,2])
        with col1:
            clear_files=st.checkbox(
                                "Remove uploaded files once no longer needed",
                                                        value=True, key=None)
        with col2:
            load_settings_from_file = st.checkbox(
                        "Load GALLOP settings from json", value=False, key=None)
        example = None
    if load_settings_from_file:
        with st.beta_expander(label="Choose settings to load", expanded=False):
            selection=st.multiselect("",settings,default=settings,key="upload")

    sdi = None
    gpx  = None
    out = None
    dbf = None
    json_settings = None
    zms = []
    if file_source == "Upload files":
        if len(uploaded_files) > 0:
            for u in uploaded_files:
                data = u.read()
                name = u.name
                name = name.replace(" ","_")
                if ".gpx" in name:
                    data = bytearray(data)
                    with open(name, "wb") as output:
                        output.write(data)
                else:
                    data = StringIO(data.decode("utf-8")).read().split("\r")
                    with open(name, "w") as output:
                        if ".sdi" not in name:
                            output.writelines(data)
                        else:
                            lines = []
                            for line in data:
                                if any(item in line for item in [
                                                "TIC","HCV","PIK","RAW","DSL"]):
                                    line = line.split(".\\")
                                    line[1] = line[1].replace(" ", "_")
                                    line = ".\\".join(line)
                                    lines.append(line)
                                else:
                                    lines.append(line)
                            output.writelines(lines)
                    output.close()
                if ".sdi" in name:
                    sdi = name
                if ".zmatrix" in name:
                    zms.append(name)
                if ".gpx" in name:
                    gpx = name
                if ".out" in name:
                    out = name
                if ".json" in name:
                    json_settings = name
                if ".dbf" in name:
                    dbf = name

    else:
        if example == "Verapamil hydrochloride":
            sdi = os.path.join("data","Verap.sdi")
            zms = [os.path.join("data","CURHOM_1.zmatrix"),
                os.path.join("data","CURHOM_2_13_tors.zmatrix")]
        elif example == "Famotidine form B":
            sdi = os.path.join("data","Famotidine.sdi")
            zms = [os.path.join("data","FOGVIG03_1.zmatrix")]

    if load_settings_from_file and json_settings is not None:
        with open(json_settings, "r") as f:
            json_settings = json.load(f)
        f.close()

        for s in settings:
            if s not in selection:
                json_settings.pop(s, None)

    return uploaded_files, sdi, gpx, out, json_settings, zms, dbf, \
            load_settings, pawley_program,clear_files





def display_info(struct, all_settings, minimiser_settings, pawley_program):
    # Display info about GALLOP and the structure in collapsible tables
    st.markdown("**GALLOP and structure information**")
    if all_settings["device"] == "Auto":
        device_name = ('cuda' if torch.cuda.is_available() else 'cpu')
        if device_name == "cuda":
            device_name = torch.cuda.get_device_name(0).split(" = ")[-1]
    else:
        device_name = all_settings["device"].split(" = ")[-1]

    col1, col2 = st.beta_columns([2,2])
    with col1:
        with st.beta_expander(label="GALLOP parameters used", expanded=False):
            gallop_info = [["Number of swarms", all_settings["n_swarms"]],
                        ["Particles per swarm", all_settings["swarm_size"]],
                        ["Total particles", (all_settings["n_swarms"]
                                                *all_settings["swarm_size"])],
                        ["Local optimiser", all_settings["optim"]],
                        ["Local optimiser hardware", device_name],]
            gallop_info = pd.DataFrame(gallop_info,
                                                columns=["Parameter", "Value"])
            gallop_info.index = [""] * len(gallop_info)
            st.table(gallop_info)
    with col2:
        with st.beta_expander(label="Data", expanded=False):
            data_info = [["Wavelength", struct.wavelength],
                ["Percentage of reflections used",
                                        all_settings["reflection_percentage"]],
                ["Number of reflections", minimiser_settings["n_reflections"]],
                ["Data resolution",
                        struct.dspacing[minimiser_settings["n_reflections"]-1]],
                ["Pawley Refinement Software", pawley_program]]
            data_info = pd.DataFrame(data_info, columns=["Parameter", "Value"])
            data_info.index = [""] * len(data_info)
            st.table(data_info)

    col1, col2 = st.beta_columns([2,2])
    with col1:
        with st.beta_expander(label="Unit Cell", expanded=False):
            space_group = struct.space_group.symbol
            cell_info = [["a", np.around(struct.lattice.a, 3)],
                        ["b", np.around(struct.lattice.b, 3)],
                        ["c", np.around(struct.lattice.c, 3)],
                        ["al", np.around(struct.lattice.alpha, 3)],
                        ["be", np.around(struct.lattice.beta, 3)],
                        ["ga", np.around(struct.lattice.gamma, 3)],
                        ["Volume", np.around(struct.lattice.volume, 3)],
                        ["Space Group", space_group],]
                        #["International number", struct.sg_number]]
            cell_info = pd.DataFrame(cell_info, columns=["Parameter", "Value"])
            cell_info.index = [""] * len(cell_info)
            st.table(cell_info)

    with col2:
        with st.beta_expander(label="Degrees of freedom", expanded=False):
            zm_info = []
            dof_tot = 0
            for zm in struct.zmatrices:
                filename, dof = zm.filename, zm.degrees_of_freedom
                if zm.rotation_degrees_of_freedom == 4:
                    dof -= 1
                zm_info.append([filename, dof])
                dof_tot+=dof
            zm_info.append(["TOTAL:", dof_tot])
            zm_info = pd.DataFrame(zm_info, columns=["Filename", "DoF"])
            zm_info.index = [""] * len(zm_info)
            st.table(zm_info)






def find_learning_rate(all_settings, minimiser_settings, struct,
                        external, internal):
    # Learning rate finder interface
    col1, col2 = st.beta_columns(2)
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
        lr = optimiser.find_learning_rate(struct, external=external[0:end],
                                internal=internal[0:end],
                                multiplication_factor=all_settings["mult"],
                                min_lr=-4, max_lr=np.log10(0.15),
                                plot=False,
                                logplot=False,
                                minimiser_settings = minimiser_settings)
        lrs = lr[0].copy()
        losses = lr[1].copy()
        lrs -= lrs.min()
        losses -= losses.min()
        lrs /= lrs.max()
        losses /= losses.max()
        with col2:
            with st.beta_expander(label=" ", expanded=False):
                minpoint = np.argmin(losses)
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 2))
                ax[0].plot(lr[0], lr[1])
                ax[1].plot(lrs[minpoint:],
                    lrs[minpoint:]-lrs[minpoint:].min(),alpha=0.5,c="k")
                ax[1].plot(lrs[minpoint:],
                    0.5*(lrs[minpoint:]-lrs[minpoint:].min()),
                    alpha=0.5,c="k")
                ax[1].plot(lrs[minpoint:],
                    0.25*(lrs[minpoint:]-lrs[minpoint:].min()),
                    alpha=0.5,c="k")
                ax[1].plot(lrs[minpoint:], losses[minpoint:])
                ax[0].set_xlabel('learning rate')
                ax[1].set_xlabel('normalised learning rate')
                ax[0].set_ylabel('loss')
                ax[1].set_ylabel('normalised loss')
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
                    minimiser_settings["learning_rate"] = lr[-1] * mult
                    st.write("Learning rate:", np.around(lr[-1] * mult,4))
                    st.write("Mult:", mult)
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




def browse_solved_zips():
    # Very basic file browser, limited to zips in the "GALLOP_results" directory
    # into which all result CIFs are placed.
    # Gives download links, and displays the time the file was last modified.
    st.sidebar.markdown("**Files**")
    if not os.path.exists("GALLOP_results"):
        st.sidebar.write("No results directory found!")
    else:
        subdirs = os.listdir("GALLOP_results")
        folders = []
        st.sidebar.write("Show results from:")
        dates = [datetime.datetime.strptime(s, "%Y-%b-%d") for s in subdirs]
        dates.sort(reverse=True)
        sorteddates = [datetime.datetime.strftime(d, "%Y-%b-%d") for d in dates]
        for i, s in enumerate(sorteddates):
            folders.append(st.sidebar.checkbox(s,value=i==0, key=i))
        for i, d in enumerate(zip(sorteddates, folders)):
            if d[1]:
                with st.beta_expander(label=d[0], expanded=True):
                    col1, col2 = st.beta_columns([1,3])
                    zipnames = []
                    for zipname in glob.iglob(os.path.join("GALLOP_results",
                                                            d[0],"*.zip")):
                        zipnames.append([zipname,
                                        os.path.getmtime(zipname)])
                    zipnames = [x[0] for x in sorted(zipnames, reverse=True,
                                                            key=lambda x: x[1])]
                    for zipname in zipnames:
                        with col2:
                            filename = zipname.split(d[0])[1].strip(
                                                            "\\").strip("/")
                            with open(zipname, "rb") as f:
                                file_bytes = f.read()
                                b64 = base64.b64encode(file_bytes).decode()
                                href = f'<a href="data:file/zip;base64,{b64}\
                                    " download=\'{filename}\'>{filename}</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            f.close()
                        with col1:
                            st.write(
                                datetime.datetime.fromtimestamp(
                                os.path.getmtime(zipname)).time().strftime(
                                                                    "%H:%M"))



def show_structure(result, Structure, all_settings, hide_H=True, interval=30):
    animation = all_settings["animate_structure"]
    if not animation:
        files.save_CIF_of_best_result(Structure, result, filename_root="plot")
        fn = glob.glob("plot*.cif")[0]
        cifs = []
        with open(fn, "r") as cif:
            lines = []
            for line in cif:
                if hide_H:
                    splitline = list(filter(
                            None,line.strip().split(" ")))
                    if splitline[0] != "H":
                        lines.append(line)
                else:
                    lines.append(line)
        cif.close()
        os.remove(fn)
        cif = "\n".join(lines)
        cifs.append(cif)
        view = py3Dmol.view()
        view.addModel(cif, "cif",
            {"doAssembly" : True,
            "normalizeAssembly":True,
            'duplicateAssemblyAtoms':True})
        view.setStyle({"stick":{'sphere':{"scale":0.15},
                                'stick':{"radius":0.25}}})
        view.addUnitCell()
        view.zoomTo()
        view.render()
        t = view.js()
        return t.startjs + "\n" + t.endjs + "\n"
    else:
        cifs = files.get_multiple_CIFs_from_trajectory(Structure, result)
        # First plot best structure for display in web app
        view = py3Dmol.view()
        view.addModel(cifs[-1], "cif",
            {"doAssembly" : True,
            "normalizeAssembly":True,
            'duplicateAssemblyAtoms':True})
        view.setStyle({'sphere':{"scale":0.15},
                        'stick':{"radius":0.25}})
        view.addUnitCell()
        view.zoomTo()
        view.render()
        t = view.js()

        html = t.startjs + "\n" + t.endjs + "\n"

        # Now save full cell animation
        view = py3Dmol.view()
        view.addModelsAsFrames("\n".join(cifs), 'cif',
                        {"doAssembly" : True,
                        "normalizeAssembly":True,
                        'duplicateAssemblyAtoms':True})
        view.animate({'loop': 'forward', 'interval': interval})
        view.setStyle({'model':0},{'sphere':{"scale":0.15},
                                    'stick':{"radius":0.25}})

        view.addUnitCell()
        view.zoomTo()
        #view.render()

        t = view.js()
        f = open(f'viz_{result["GALLOP Iter"]+1}_anim.html', 'w')
        f.write(t.startjs)
        f.write(t.endjs)
        f.close()

        # Now plot asymmetric unit animation
        view = py3Dmol.view()
        view.addModelsAsFrames("\n".join(cifs), 'cif',
                        {"doAssembly" : False,
                        "normalizeAssembly":True,
                        'duplicateAssemblyAtoms':True})
        view.animate({'loop': 'forward', 'interval': interval})
        view.setStyle({'model':0},{'sphere':{"scale":0.15},
                                    'stick':{"radius":0.25}})

        view.zoomTo()
        #view.render()

        t = view.js()
        f = open(f'viz_{result["GALLOP Iter"]+1}_asym_anim.html', 'w')
        f.write(t.startjs)
        f.write(t.endjs)
        f.close()

        return html