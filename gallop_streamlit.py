# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
GALLOP web app, built with Streamlit.
"""

import time
import os
import json
import torch
import streamlit as st
import numpy as np
import pandas as pd
from gallop.optim.local import minimise
from gallop import streamlit_utils as su
from gallop import optim
from gallop import structure
from gallop import multiGPU

st.set_page_config(page_title='GALLOP Web App', page_icon = ":gem:",
                    layout = 'centered', initial_sidebar_state = 'auto')

# Top section
st.title("GALLOP")
st.sidebar.title("Options")
function = st.sidebar.radio("Choose function",
                            ["GALLOP","View previous results",])
# Re-enable this in a future update, once help files are written!
# "Instructions"])

if "View" in function:
    su.browse_solved_zips()

#elif "Instructions" in function:
#    su.instructions()

elif function == "GALLOP":
    # First get the settings for the runs
    st.sidebar.markdown("**Settings**")
    all_settings = su.sidebar()

    # Now we upload the files needed for GALLOP - DASH fit files and Z-matrices
    uploaded_files, sdi, gpx, out, hkl, ins, cif, json_settings, zms, \
        dbf, load_settings_from_file, pawley_program, clear_files, \
        use_profile, step = su.files.get_files()

    st.text("")
    st.text("")
    all_files=st.button("Solve")

    if all_files and len(zms) == 0:
        st.write("No files uploaded. Upload files or select from examples")
    elif all_files and len(zms) > 0:
        if load_settings_from_file and json_settings is not None:
            all_settings.update(json_settings)

        st.text("")
        st.text("")
        # All files prepared, now need to construct the GALLOP structure object
        # and get the settings for the GALLOP runs.

        # If the structure name hasn't been changed, use the Pawley files to
        # give it a meaningful name
        all_settings["structure_name"] = su.structure.get_name(all_settings,
                                            pawley_program, sdi, gpx, out,
                                            cif, ins)
        struct = structure.Structure(
                                name=all_settings["structure_name"],
                                ignore_H_atoms=all_settings["ignore_H_atoms"])
        if all_settings["temperature"] > 0.0:
            struct.temperature = all_settings["temperature"]

        su.structure.add_data(struct, all_settings, pawley_program, sdi, gpx, out, cif,
                    ins, hkl)

        su.structure.add_zmatrices(struct, zms, all_settings)

        minimiser_settings = su.structure.get_minimiser_settings(struct,
                                            all_settings, use_profile, step)

        if all_settings["memory_opt"]:
            su.improve_GPU_memory_use(struct, minimiser_settings)
            st.write("Attempting to reduce GPU memory use at the expense of "
                    "reduced Local Optimisation speed")

        # Create the swarm object and generate initial positions for the particles
        swarm = su.structure.get_swarm(struct, all_settings)
        external, internal = swarm.get_initial_positions(MDB=dbf)
        external = np.array(external)
        internal = np.array(internal)
        failed = False

        su.files.remove_uploaded_files(clear_files, uploaded_files)

        su.display_info(struct, all_settings, minimiser_settings,
                                                    pawley_program)

        st.write("")
        st.write("")
        st.markdown("**Progress**")
        if all_settings["find_lr"]:
            all_settings, minimiser_settings, failed = su.structure.find_learning_rate(
                                                            all_settings,
                                                            minimiser_settings,
                                                            struct,
                                                            external,
                                                            internal)
        else:
            minimiser_settings["learning_rate"] = all_settings["lr"]

        if not failed:
            # Now dump all of the settings used to a JSON so it can be inspected
            # or read in at a later stage. This will be included with the zip
            # that the user downloads to get their CIFs.
            settings_file = struct.name+"_GALLOP_Settings.json"
            with open(os.path.join(os.getcwd(),settings_file), "w") as f:
                json.dump(all_settings, f, indent=4)
            f.close()

            # Outer loop over the number of independent runs requested
            nruns = int(all_settings["n_GALLOP_runs"])
            for run in range(nruns):
                if run > 0:
                    # New swarm and starting positions for each independent run
                    swarm = su.structure.get_swarm(struct, all_settings)
                    external, internal = swarm.get_initial_positions(MDB=dbf)
                    external = np.array(external)
                    internal = np.array(internal)
                st.write("")
                result_info = []
                start_time = time.time()
                run_placeholder = st.empty() # Which run is currently active
                iter_placeholder = st.empty() # Which iteration of run N
                progress_bar_placeholder = st.empty()
                result_placeholder = st.empty()
                lr = minimiser_settings["learning_rate"]

                GPU_split = all_settings["particle_division"]
                n_GPUs = torch.cuda.device_count()
                if (GPU_split is not None and n_GPUs >= len(GPU_split)):
                    import torch.multiprocessing as mp
                    mp.set_start_method('spawn', force=True)# For use with CUDA on
                                                            # Unix systems
                    pool = mp.Pool(processes = len(GPU_split))

                with run_placeholder:
                    if run+1 == nruns:
                        st.write(f"Run {run+1} of {nruns}.")
                    else:
                        st.write(f"Run {run+1} of {nruns}...")

                zipname, filename = su.files.get_zipname(struct, all_settings, run)

                # Inner loop over GALLOP iterations
                for i in range(int(all_settings["n_GALLOP_iters"])):
                    itertext = su.get_itertext(i, all_settings)
                    iter_placeholder.text(itertext)
                    # Run the local optimisation component of GALLOP
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

                    # Now display some output to the user, including a link to
                    # download the CIFs
                    col1, col2 = result_placeholder.columns([1,2])
                    with col1:
                        # Zip and then delete the cifs, then make a download
                        # link for the zip file
                        su.files.get_download_link(run, i, zipname, result,
                                                all_settings, filename, struct)
                        st.table(result_info_df.iloc[::-1])
                    with col2:
                        swarm_plot, structure_plot, profile_plot = st.tabs([
                            "Swarms", "View structure", "View Profile"])
                        # Show the swarms
                        with swarm_plot:
                            su.plots.plot_swarms(chi_2, all_settings)
                        # Display the structure
                        with structure_plot:
                            su.plots.plot_structure(result, struct, all_settings, i)
                        # If using DASH data, plot the diffraction data
                        with profile_plot:
                            su.plots.plot_profile(struct, result, i)

                    # Perform the PSO update and get the new starting points
                    external, internal = swarm.update_position(result=result,
                                                                verbose=False)

                    # Now optionally randomise the worst performing particles if
                    # requested by the user.
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
                    # If using torsion shadowing, check if the number of assigned
                    # iterations has passed, and if so, disable shadowing.
                    if ((i+1)==all_settings["shadow_iters"] and
                                                all_settings["torsion_shadowing"]):
                        minimiser_settings["torsion_shadowing"] = False
                        # Add a little random noise to the internal DoF as they
                        # will all be the same. This is roughly +/- 10 deg
                        internal += np.random.uniform(-0.157,0.157,
                                        size=internal.shape)

                if (GPU_split is not None and n_GPUs >= len(GPU_split)):
                    pool.close()
                    pool.join()
