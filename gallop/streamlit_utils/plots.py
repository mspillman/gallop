# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for the web app focussed on plotting
"""
import numpy as np
import streamlit as st
import mpld3
import matplotlib.pyplot as plt
import py3Dmol
import pandas as pd
import altair as alt
import glob
import os

from gallop import files


def plot_swarms(chi_2, all_settings):
    st.write("")
    st.write("")
    labels = np.ones_like(chi_2)
    for j in range(all_settings["n_swarms"]):
        begin = j*all_settings["swarm_size"]
        end = (j+1)*all_settings["swarm_size"]
        labels[begin:end] *= j
        labels[begin:end] += 1
    chi2_info = pd.DataFrame({
        "chi^2" : chi_2,
        "swarm" : labels
    })
    alt.data_transformers.disable_max_rows()

    chart = alt.layer(alt.Chart(chi2_info).mark_tick().encode(
        x='chi^2:Q',
        y='swarm:O'
    )).interactive()

    st.altair_chart(chart)

def plot_structure(result, Structure, all_settings, i, hide_H=True, interval=30):
    if hide_H:
        st.write(f"Iteration {i+1}. H hidden for clarity")
    else:
        st.write(f"Iteration {i+1}")
    animation = all_settings["animate_structure"]
    if not animation:
        files.save_CIF_of_best_result(Structure, result, filename_root="plot")
        fn = glob.glob("plot*.cif")[0]
        with open(fn, "r") as cif:
            lines = []
            for line in cif:
                if hide_H:
                    splitline = list(filter(None,line.strip().split(" ")))
                    if splitline[0] != "H":
                        lines.append(line)
                else:
                    lines.append(line)
        cif.close()
        os.remove(fn)
        cif = "\n".join(lines)
    else:
        cifs = files.get_multiple_CIFs_from_trajectory(Structure, result,
                                                        decimals=3)
        files.save_animation_from_trajectory(result, Structure, cifs=cifs,
            filename_root="plot", interval=interval)
        cif = cifs[-1]
    view = py3Dmol.view()
    view.addModel(cif, "cif",
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

    st.components.v1.html(html, width=600, height=400)


def plot_profile(struct, result, i):
    if struct.source.lower() == "dash":
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
                                figsize=(6,4))

        ax[0].plot(struct.profile[:,0], struct.profile[:,1])
        ax[0].plot(struct.profile[:,0], result["calc_profile"])
        ax[1].plot(struct.profile[:,0], (struct.profile[:,1]
            - result["calc_profile"])/struct.profile[:,2])

        ax[0].title.set_text(f"Iteration {i+1}, "
                    +"Profile χ² = "+str(
                    np.around(result["prof_chi_2"], 3)))
        ax[0].set_ylabel('Intensity')
        ax[0].legend(["Obs", "Calc"])
        ax[1].set_xlabel('2θ')
        ax[1].set_ylabel('ΔI/σ')
        #st.pyplot(fig)
        fig_html = mpld3.fig_to_html(fig)
        st.components.v1.html(fig_html, width=600, height=450)
    else:
        st.write("Profile plotting is currently only sup"
                "ported with DASH data")