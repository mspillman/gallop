# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for the web app
"""

# Others needed
import glob
import base64
import datetime
import os
import sys
import torch
import streamlit as st
import pandas as pd
import numpy as np
from collections import OrderedDict

# GALLOP related
from .. import chi2
from .. import optim
from .. import tensor_prep

from . import plots
from . import settings
from . import files
from . import structure



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
    swarm = optim.swarm.Swarm(Structure=struct, n_particles=1, n_swarms=1)
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


def sidebar():
    all_settings, filename = settings.load_settings()

    all_settings = settings.get_all_settings(all_settings)

    settings.save_settings(all_settings, filename)

    st.sidebar.write("")
    reset = st.sidebar.button("Reset GALLOP")
    # This code for resetting streamlit is a bit of an ugly hack, but it
    # works. If there's a more elegant way to do it, it's not easy to find!
    # The code writes out a short script - batch on Windows, otherwise a .py
    # which is then called. This kills the running streamlit process, and then
    # starts a new one.
    # Need parent directory of GALLOP folder to check if it's been run from a
    # file, or if GALLOP is installed.
    if reset:
        filedir = os.path.dirname(sys.argv[0])
        print(filedir)
        # Windows
        if os.name == "nt":
            lines = ["taskkill /IM \"streamlit.exe\" /F /T\n",
                    "streamlit run " + os.path.join(filedir,
                                                    "gallop_streamlit.py")]
            with open(os.path.join(filedir,"reset.bat"), "w") as reset_script:
                reset_script.writelines(lines)
            reset_script.close()
            os.system("START /B "+os.path.join(filedir,"reset.bat"))
        # Linux
        else:
            script = str(os.path.join(filedir,"gallop_streamlit.py"))
            lines = ["import os\n",
                "import time\n",
                "os.system(\"pkill -f streamlit\")\n",
                "time.sleep(1)\n",
                f"# comment {script}\n",
                f"os.system(\"streamlit run {script}\")\n",]
            with open(os.path.join(filedir,"reset.py"), "w") as reset_script:
                reset_script.writelines(lines)
            reset_script.close()
            os.system("python3 "+os.path.join(filedir,"reset.py")+" &>/dev/null")

    return all_settings




def display_info(struct, all_settings, minimiser_settings, pawley_program):
    # Display info about GALLOP and the structure in collapsible tables
    st.markdown("**GALLOP and structure information**")
    if all_settings["device"] == "Auto":
        device_name = ('cuda' if torch.cuda.is_available() else 'cpu')
        if device_name == "cuda":
            device_name = torch.cuda.get_device_name(0).split(" = ")[-1]
    else:
        device_name = all_settings["device"].split(" = ")[-1]

    col1, col2 = st.columns([2,2])
    with col1:
        with st.expander(label="GALLOP parameters used", expanded=False):
            gallop_info = [["Number of swarms", str(all_settings["n_swarms"])],
                        ["Particles per swarm", str(all_settings["swarm_size"])],
                        ["Total particles", str(all_settings["n_swarms"]
                                                *all_settings["swarm_size"])],
                        ["Local optimiser", all_settings["optim"]],
                        ["Local optimiser hardware", device_name],]
            gallop_info = pd.DataFrame(gallop_info,
                                                columns=["Parameter", "Value"])
            gallop_info.index = [""] * len(gallop_info)
            st.table(gallop_info)
    with col2:
        with st.expander(label="Data", expanded=False):
            data_info = [
                ["Wavelength", str(struct.wavelength)],
                ["Percentage of reflections used",
                                str(all_settings["reflection_percentage"])],
                ["Number of reflections", 
                                str(minimiser_settings["n_reflections"])],
                ["Data resolution",
                    str(struct.dspacing[minimiser_settings["n_reflections"]-1])],
                ["Pawley Refinement Software", pawley_program]
                ]
            data_info = pd.DataFrame(data_info, columns=["Parameter", "Value"])
            data_info.index = [""] * len(data_info)
            st.table(data_info)

    col1, col2 = st.columns([2,2])
    with col1:
        with st.expander(label="Unit Cell", expanded=False):
            space_group = struct.space_group.symbol
            cell_info = [["a", str(np.around(struct.lattice.a, 3))],
                        ["b",  str(np.around(struct.lattice.b, 3))],
                        ["c",  str(np.around(struct.lattice.c, 3))],
                        ["al", str(np.around(struct.lattice.alpha, 3))],
                        ["be", str(np.around(struct.lattice.beta, 3))],
                        ["ga", str(np.around(struct.lattice.gamma, 3))],
                        ["Volume", str(np.around(struct.lattice.volume, 3))],
                        ["Space Group", space_group],]
                        #["International number", struct.sg_number]]
            cell_info = pd.DataFrame(cell_info, columns=["Parameter", "Value"])
            cell_info.index = [""] * len(cell_info)
            st.table(cell_info)

    with col2:
        with st.expander(label="Degrees of freedom", expanded=False):
            zm_info = []
            dof_tot = 0
            for zm in struct.zmatrices:
                filename = os.path.split(zm.filename)[-1]
                dof = zm.degrees_of_freedom
                if zm.rotation_degrees_of_freedom == 4:
                    dof -= 1
                zm_info.append([filename, str(dof)])
                dof_tot+=dof
            zm_info.append(["TOTAL:", str(dof_tot)])
            zm_info = pd.DataFrame(zm_info, columns=["Filename", "DoF"])
            zm_info.index = [""] * len(zm_info)
            st.table(zm_info)


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
        yearmonths = OrderedDict()
        for i, s in enumerate(sorteddates):
            yearmonth = "-".join(s.split("-")[:2])
            if yearmonth not in yearmonths:
                yearmonths[yearmonth] = [s]
            else:
                yearmonths[yearmonth].append(s)
        all_vals = 0
        for i, item in enumerate(yearmonths.items()):
            k, v = item
            with st.sidebar.expander(label=k, expanded=i==0):
                for j, s in enumerate(v):
                    folders.append(st.checkbox(s,value=(i==0 and j==0), key=str(all_vals)))
                    all_vals += 1

        for i, d in enumerate(zip(sorteddates, folders)):
            if d[1]:
                with st.expander(label=d[0], expanded=True):
                    col1, col2 = st.columns([1,3])
                    zipnames = []
                    for zipname in glob.iglob(os.path.join("GALLOP_results",
                                                            d[0],"*.zip")):
                        zipnames.append([zipname,
                                        os.path.getmtime(zipname)])
                    zipnames = [x[0] for x in sorted(zipnames, reverse=True,
                                                    key=lambda x: str(x[1]))]
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


def instructions():
    with open("./help.md") as readme:
        lines = readme.readlines()
    readme.close()
    lines = "".join(lines)
    st.video('https://www.youtube.com/watch?v=n0aovGUS4JU')
    st.markdown(lines, unsafe_allow_html=True)


def get_itertext(i, all_settings):
    itertext = "GALLOP iteration " + str(i+1)
    if ((i+1)%all_settings["global_update_freq"] == 0
            and i != 0 and all_settings["global_update"]):
        itertext += " Global update after this iter"
    if ((i+1)==all_settings["shadow_iters"] and
                                all_settings["torsion_shadowing"]):
        itertext += " Removing torsion shadowing after this iter"
    if all_settings["randomise_worst"]:
        if (i+1) % all_settings["randomise_freq"] == 0:
            pcnt = all_settings["randomise_percentage"]
            itertext += f" Randomising worst {pcnt} % of particles"
    return itertext