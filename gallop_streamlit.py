# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
GALLOP web app, built with Streamlit.
"""

import time
import os
import glob
import base64
import json
import datetime
from zipfile import ZipFile, ZIP_DEFLATED
import torch
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import gallop_streamlit_utils as gsu
from gallop import z_matrix
from gallop import optim
from gallop import structure
from gallop import multiGPU

st.set_page_config(page_title='GALLOP Web App', page_icon = ":gem:",
                    layout = 'centered', initial_sidebar_state = 'auto')



# Top section
st.title("GALLOP")
st.sidebar.title("Options")
function = st.sidebar.radio("Choose function",
                            ["GALLOP","View previous results","Instructions"])

if "View" in function:
    gsu.browse_solved_zips()

elif "Instructions" in function:
    gsu.instructions()

elif function == "GALLOP":
    st.sidebar.markdown("**Settings**")
    all_settings = gsu.sidebar()

    # Now we upload the files needed for GALLOP - DASH fit files and Z-matrices
    uploaded_files, sdi, gpx, out, json_settings, zms, dbf, load_settings, \
                                pawley_program, clear_files = gsu.get_files()


    st.text("")
    st.text("")
    all_files=st.button("Solve")

    if all_files and len(zms) == 0:
        st.write("No files uploaded. Upload files or select from examples")
    elif all_files and len(zms) > 0:
        if load_settings and json_settings is not None:
            all_settings.update(json_settings)

        st.text("")
        st.text("")
        # All files prepared, now need to construct the GALLOP structure object
        # and get the settings for the GALLOP runs.

        # If the structure name hasn't been changed, use the Pawley files to
        # give it a meaningful name
        name = all_settings["structure_name"]
        if name == "Enter_structure_name" or len(name) == 0:
            if pawley_program == "DASH":
                structure_name = os.path.split(sdi)[-1].split(".sdi")[0]
            elif pawley_program == "GSAS-II":
                structure_name = os.path.split(gpx)[-1].split(".gpx")[0]
            else:
                structure_name = os.path.split(out)[-1].split(".out")[0]
            all_settings["structure_name"] = structure_name
        struct = structure.Structure(
                                name=all_settings["structure_name"],
                                ignore_H_atoms=all_settings["ignore_H_atoms"])
        if pawley_program == "DASH":
            struct.add_data(sdi, source="DASH",
                        percentage_cutoff=all_settings["percentage_cutoff"])
        elif pawley_program == "GSAS-II":
            struct.add_data(gpx, source="GSAS",
                        percentage_cutoff=all_settings["percentage_cutoff"])
        else:
            struct.add_data(out, source="TOPAS",
                        percentage_cutoff=all_settings["percentage_cutoff"])
        for z in zms:
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
        if all_settings["temperature"] > 0.0:
            struct.temperature = all_settings["temperature"]
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

        if all_settings["use_restraints"]:
            if all_settings["restraints"] is not None:
                for r in all_settings["restraints"].keys():
                    r = all_settings["restraints"][r]
                    r = r.replace(" ","").split(",")
                    struct.add_restraint(atom1=r[0], atom2=r[1],
                        distance=float(r[2]), percentage=float(r[3]))
            minimiser_settings["use_restraints"] = True
        if all_settings["animate_structure"]:
            minimiser_settings["save_trajectories"] = True
        if all_settings["memory_opt"]:
            gsu.improve_GPU_memory_use(struct, minimiser_settings)
            st.write("Attempting to reduce GPU memory use at the expense of "
                    "reduced Local Optimisation speed")
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

        external, internal = swarm.get_initial_positions(MDB=dbf)
        external = np.array(external)
        internal = np.array(internal)
        failed = False

        if clear_files:
            for u in uploaded_files:
                u.read()
                name = u.name.replace(" ", "_")
                os.remove(name)

        gsu.display_info(struct, all_settings, minimiser_settings,
                                                    pawley_program)

        st.write("")
        st.write("")
        st.markdown("**Progress**")
        if all_settings["find_lr"]:
            all_settings, minimiser_settings, failed = gsu.find_learning_rate(
                                                            all_settings,
                                                            minimiser_settings,
                                                            struct,
                                                            external,
                                                            internal)
        else:
            minimiser_settings["learning_rate"] = all_settings["lr"]

        if not failed:
            # Now dump all of the settings used to a JSON so it can be inspected
            # or read in at a later stage.
            settings_file = struct.name+"_GALLOP_Settings.json"
            with open(os.path.join(os.getcwd(),settings_file), "w") as f:
                json.dump(all_settings, f, indent=4)
            f.close()

            st.write("")
            st.write("Solving...")
            result_info = []
            start_time = time.time()
            iter_placeholder = st.empty()
            progress_bar_placeholder = st.empty()
            structure_plot_placeholder = st.empty()
            if struct.source.lower() == "dash":
                profile_plot_placeholder = st.empty()
            result_placeholder = st.empty()

            now = datetime.datetime.now()
            current_time = str(now.strftime("%H-%M"))

            lr = minimiser_settings["learning_rate"]
            zipname = f'{struct.name}_{current_time}_{all_settings["n_swarms"]}\
                        _swarms_{all_settings["swarm_size"]}\
                        _particles_{all_settings["optim"]}.zip'
            zipname = zipname.replace(" ","")
            if not os.path.exists("GALLOP_results"):
                os.mkdir("GALLOP_results")

            date = datetime.date.today().strftime("%Y-%b-%d")
            if not os.path.exists(os.path.join("GALLOP_results", date)):
                os.mkdir(os.path.join("GALLOP_results", date))
            zipname = os.path.join("GALLOP_results",date,zipname)

            GPU_split = all_settings["particle_division"]
            n_GPUs = torch.cuda.device_count()
            if (GPU_split is not None and n_GPUs >= len(GPU_split)):
                import torch.multiprocessing as mp
                mp.set_start_method('spawn', force=True)# For use with CUDA on
                                                        # Unix systems
                pool = mp.Pool(processes = len(GPU_split))

            for i in range(int(all_settings["n_GALLOP_iters"])):
                itertext = "GALLOP iteration " + str(i+1)
                if ((i+1)%all_settings["global_update_freq"] == 0 and i != 0 and
                                                all_settings["global_update"]):
                    itertext += " Global update after this iter"
                if ((i+1)==all_settings["shadow_iters"] and
                                            all_settings["torsion_shadowing"]):
                    itertext += " Removing torsion shadowing after this iter"
                if all_settings["randomise_worst"]:
                    if (i+1) % all_settings["randomise_freq"] == 0:
                        pcnt = all_settings["randomise_percentage"]
                        itertext += f" Randomising worst {pcnt} % of particles"

                iter_placeholder.text(itertext)
                with progress_bar_placeholder:
                    try:
                        if (GPU_split is not None and n_GPUs >= len(GPU_split)):
                            result = multiGPU.minimise(i, struct, swarm,
                                external, internal, GPU_split,
                                minimiser_settings, pool, start_time=start_time)

                        else:
                            result = optim.local.minimise(struct,
                                                external=external,
                                                internal=internal,
                                                run=i, start_time=start_time,
                                                **minimiser_settings)
                    except RuntimeError as e:
                        if "memory" in str(e):
                            st.error("GPU memory error! Reset GALLOP, then:")
                            st.write("Try reducing number of particles, the "
                                    "number of reflections or use the Reduce "
                                    "performance option in Local optimiser "
                                    "settings.")
                            st.write("Error code below:")
                            st.write("")
                            st.text(str(e))
                        else:
                            st.error("An unknown error occurred:")
                            st.text(str(e))
                        break
                chi_2 = result["chi_2"]

                result_info.append([chi_2.min().item(),
                                (time.time() - start_time)/60])
                result_info_df = pd.DataFrame(result_info, columns=["best chi2",
                                                                "time / min"])
                result_info_df.index = np.arange(1, len(result_info_df) + 1)
                with structure_plot_placeholder:
                    hide_H = True
                    with st.expander(label="Show structure", expanded=False):
                        html = gsu.show_structure(result, struct, all_settings,
                                                            hide_H=hide_H)
                        #st.components.v1.html(open(
                        ##    f'viz_{result["GALLOP Iter"]+1}.html', 'r').read(),
                        #                        width=600, height=400)
                        st.components.v1.html(html, width=600, height=400)
                        if hide_H:
                            st.write(f"Iteration {i+1}. H hidden for clarity")
                        else:
                            st.write(f"Iteration {i+1}")

                if struct.source.lower() == "dash":
                    with profile_plot_placeholder:
                        with st.expander(label="Show profile",
                                                                expanded=False):
                            if struct.ignore_H_atoms:
                                st.write("Stats with H-atoms included:")
                            st.write("$\\chi^{2}_{int}$ = "+str(
                                    np.around(result["best_chi_2_with_H"], 3)))
                            st.write("$\\chi^{2}_{prof}$ = "+str(
                                            np.around(result["prof_chi_2"], 3)))
                            ratio = np.around(
                                result["prof_chi_2"] / struct.PawleyChiSq, 3)
                            st.write("$\\frac{\\chi^{2}_{prof}}"
                                    + "{\\chi^{2}_{Pawley}}$ ="
                                    + " " + str(ratio))

                            fig, ax = plt.subplots(2, 1, gridspec_kw={
                                                    'height_ratios': [4, 1]},
                                                    figsize=(10,8))
                            ax[0].plot(struct.profile[:,0], struct.profile[:,1])
                            ax[0].plot(struct.profile[:,0], result["calc_profile"])
                            ax[1].plot(struct.profile[:,0], (struct.profile[:,1]
                                - result["calc_profile"])/struct.profile[:,2])
                            #ax[0].set_xlabel('2$\\theta$')
                            ax[0].title.set_text(f"Iteration {i+1}, "
                                        +"$\\chi^{2}_{prof}$ = "+str(
                                        np.around(result["prof_chi_2"], 3)))
                            ax[0].set_ylabel('Intensity')
                            ax[0].legend(["Obs", "Calc"])
                            ax[1].set_xlabel('2$\\theta$')
                            ax[1].set_ylabel('$\\Delta(I)/\\sigma(I_{obs})$')
                            st.pyplot(fig)

                col1, col2 = result_placeholder.columns([2,2])
                with col1:
                    # Zip and then delete the cifs, then download the zip
                    if i == 0:
                        zipObj = ZipFile(zipname, 'w')
                        settings_file = struct.name+"_GALLOP_Settings.json"
                        zipObj.write(settings_file)
                        os.remove(settings_file)

                    else:
                        zipObj = ZipFile(zipname, 'a', ZIP_DEFLATED)
                    if all_settings["animate_structure"]:
                        html = f'viz_{result["GALLOP Iter"]+1}_anim.html'
                        zipObj.write(html)
                        os.remove(html)
                        html = f'viz_{result["GALLOP Iter"]+1}_asym_anim.html'
                        zipObj.write(html)
                        os.remove(html)
                    for fn in glob.iglob("*_chisqd_*"):
                        zipObj.write(fn)
                        os.remove(fn)
                    zipObj.close()
                    filename = zipname.split(date)[1].strip("\\").strip("_")
                    with open(os.path.join(os.getcwd(), zipname), "rb") as f:
                        file_bytes = f.read()
                        b64 = base64.b64encode(file_bytes).decode()
                        href = f'<a href="data:file/zip;base64,{b64}\
                                " download=\'{filename}\'>\
                                Click for CIFs</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    f.close()

                    st.table(result_info_df.iloc[::-1])
                with col2:
                    st.write("")
                    st.write("")
                    labels = np.ones_like(chi_2)
                    for j in range(all_settings["n_swarms"]):
                        begin = j*all_settings["swarm_size"]
                        end = (j+1)*all_settings["swarm_size"]
                        labels[begin:end] *= j
                        labels[begin:end] += 1
                    chi2_info = pd.DataFrame({
                        "chi2" : chi_2,
                        "swarm" : labels
                    })
                    alt.data_transformers.disable_max_rows()

                    chart = alt.layer(alt.Chart(chi2_info).mark_tick().encode(
                        x='chi2:Q',
                        y='swarm:O'
                    )).interactive()

                    st.altair_chart(chart)
                external, internal = swarm.update_position(result=result,
                                                            verbose=False)

                if all_settings["randomise_worst"]:
                    if (i+1) % all_settings["randomise_freq"] == 0:
                        pcnt = all_settings["randomise_percentage"] / 100.
                        to_randomise = swarm.best_chi_2 >= np.percentile(
                                                    swarm.best_chi_2, 100.-pcnt)
                        external[to_randomise] = np.random.uniform(-1, 1,
                                        size=external[to_randomise].shape)
                        internal[to_randomise] = np.random.uniform(-np.pi,
                                    np.pi,size=internal[to_randomise].shape)
                        swarm.best_chi_2[to_randomise] = np.inf
                        swarm.velocity[to_randomise] *= 0

                if ((i+1)==all_settings["shadow_iters"] and
                                            all_settings["torsion_shadowing"]):
                    minimiser_settings["torsion_shadowing"] = False
                    # Add a little random noise to the internal DoF as they will
                    # all be the same. This is roughly +/- 10 deg
                    internal += np.random.uniform(-0.157,0.157,
                                    size=internal.shape)

            if (GPU_split is not None and n_GPUs >= len(GPU_split)):
                pool.close()
                pool.join()
