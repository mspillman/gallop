# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for the web app focussed on files
"""
import streamlit as st
from io import StringIO
import os
import gallop
import json
import datetime
from zipfile import ZipFile, ZIP_DEFLATED
import base64
import glob

settings_labels = ["structure_name", "n_GALLOP_iters", "seed",
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

def get_files():
    # Upload files using uploader or select from examples
    col1, col2 = st.columns(2)
    with col1:
        file_source = st.radio("Upload files or select from examples",
                                        ["Upload files","Use examples"])
    if file_source == "Use examples":
        pawley_program = "DASH"
        example = st.selectbox("Select example structure",
                    ["Famotidine form B", "Verapamil hydrochloride"])
        uploaded_files = []
        clear_files = False
        load_settings_from_file = False
    else:
        with col2:
            pawley_program = st.radio("Choose Pawley refinement program",
                                    ["DASH","GSAS-II", "TOPAS (experimental)",
                                    "SHELX (experimental)"])
            st.write("")
            st.write("")
            st.write("")
            st.write("")
    if pawley_program.lower() == "dash":
        with col1:
            use_profile = st.checkbox("Optimise using profile χ²", False)
            if use_profile:
                step = st.number_input("Profile step size", min_value=1, value=1)
            else:
                step = 1
    else:
        use_profile = False
        step = 1
    if file_source == "Upload files":
        if pawley_program == "DASH":
            uploaded_files = st.file_uploader("Upload DASH Pawley files and \
                                    Z-matrices",
                                    accept_multiple_files=True,
                                    type=["zmatrix", "sdi", "hcv", "tic", "dsl",
                                        "pik", "dbf", "json"],
                                    key=None)
        elif pawley_program == "GSAS-II":
            uploaded_files = st.file_uploader("Upload GSAS-II project file and\
                                    Z-matrices",
                                    accept_multiple_files=True,
                                    type=["zmatrix", "gpx", "json"],
                                    key=None)
        elif pawley_program == "TOPAS (experimental)":
            uploaded_files = st.file_uploader("Upload TOPAS .out file and\
                                    Z-matrices",
                                    accept_multiple_files=True,
                                    type=["zmatrix", "out", "json"],
                                    key=None)
        else:
            uploaded_files = st.file_uploader("Upload SHELX .hkl file in HKLF 4\
                                    format and Z-matrices, plus either .ins or \
                                    .cif containing cell and space group info",
                                    accept_multiple_files=True,
                                    type=["zmatrix","ins","hkl","cif","json"],
                                    key=None)
        col1, col2 = st.columns([2,2])
        with col1:
            clear_files=st.checkbox(
                                "Remove uploaded files once no longer needed",
                                                        value=True, key=None)
        with col2:
            load_settings_from_file = st.checkbox(
                        "Load GALLOP settings from json", value=False, key=None)
        example = None

    if load_settings_from_file:
        with st.expander(label="Choose settings to load", expanded=False):
            selection=st.multiselect("",settings_labels,default=settings_labels,key="upload")

    sdi = None
    gpx  = None
    out = None
    dbf = None
    hkl = None
    ins = None
    cif = None
    json_settings = None
    zms = []
    if file_source == "Upload files":
        if len(uploaded_files) > 0:
            for u in uploaded_files:
                data = u.read()
                name = u.name
                if ".gpx" in name:
                    data = bytearray(data)
                    with open(name, "wb") as output:
                        output.write(data)
                else:
                    data = StringIO(data.decode("utf-8")).read().split("\r")
                    with open(name, "w") as output:
                        output.writelines(data)
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
                if ".hkl" in name:
                    hkl = name
                if ".ins" in name:
                    ins = name
                if ".cif" in name:
                    cif = name

    else:
        filedir = os.path.dirname(gallop.__file__)
        if example == "Verapamil hydrochloride":
            sdi = os.path.join(filedir,"example_data","Verap.sdi")
            zms = [os.path.join(filedir, "example_data","CURHOM_1.zmatrix"),
            os.path.join(filedir, "example_data","CURHOM_2_13_tors.zmatrix")]
        elif example == "Famotidine form B":
            sdi = os.path.join(filedir, "example_data","Famotidine.sdi")
            zms = [os.path.join(filedir,"example_data","FOGVIG03_1.zmatrix")]

    if load_settings_from_file and json_settings is not None:
        with open(json_settings, "r") as f:
            json_settings = json.load(f)
        f.close()

        for s in settings_labels:
            if s not in selection:
                json_settings.pop(s, None)

    return [uploaded_files, sdi, gpx, out, hkl, ins, cif, json_settings, zms,
            dbf, load_settings_from_file, pawley_program, clear_files,
            use_profile, step]


def remove_uploaded_files(clear_files, uploaded_files):
    if clear_files:
        for u in uploaded_files:
            u.read()
            try:
                os.remove(u.name)
            except FileNotFoundError:
                name = u.name.replace(" ", "_")
                try:
                    os.remove(name)
                except FileNotFoundError:
                    print("Failed to remove file", name)

def get_zipname(struct, all_settings, run):
    now = datetime.datetime.now()
    current_time = str(now.strftime("%H-%M"))
    zipname = f'{struct.name}_run_{run+1}_{current_time}_\
            {all_settings["n_swarms"]}_swarms_\
            {all_settings["swarm_size"]}_particles_\
            {all_settings["optim"]}.zip'
    zipname = zipname.replace(" ","")
    if not os.path.exists("GALLOP_results"):
        os.mkdir("GALLOP_results")

    date = datetime.date.today().strftime("%Y-%b-%d")
    if not os.path.exists(os.path.join("GALLOP_results", date)):
        os.mkdir(os.path.join("GALLOP_results", date))
    zipname = os.path.join("GALLOP_results",date,zipname)
    filename = zipname.split(date)[1].strip("\\").strip("_")
    return zipname, filename

def get_download_link(run, i, zipname, result, all_settings, filename, struct):
    if i == 0 and run == 0:
        zipObj = ZipFile(zipname, 'w')
        settings_file = struct.name+"_GALLOP_Settings.json"
        zipObj.write(settings_file)
        os.remove(settings_file)
    else:
        zipObj = ZipFile(zipname, 'a', ZIP_DEFLATED)
    if all_settings["animate_structure"]:
        html = f'plot_iter_{result["GALLOP Iter"]+1}_both_anim.html'
        zipObj.write(html)
        os.remove(html)

    for fn in glob.iglob("*_chisqd_*"):
        zipObj.write(fn)
        os.remove(fn)
    zipObj.close()

    with open(os.path.join(os.getcwd(), zipname), "rb") as f:
        file_bytes = f.read()
        b64 = base64.b64encode(file_bytes).decode()
        href = f'<a href="data:file/zip;base64,{b64}\
                " download=\'{filename}\'>\
                CIFs for run {run+1}</a>'
        st.markdown(href, unsafe_allow_html=True)
    f.close()