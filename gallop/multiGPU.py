# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Allows GALLOP to run on multiple GPUs via multiprocessing.
This may be slower than simply running multiple instances of GALLOP
simultaneously, so not really recommended unless needed.
"""

import torch
import multiprocessing as mp
import numpy as np
import time
from gallop import structure
from gallop import optimiser
from gallop import files

def multiGPU(GPU, start, end, external, internal, structure_files,
            minimiser_settings):
    """
    Process GALLLOP jobs on multiple GPUs

    Args:
        GPU (int): The GPU to use. Typically, this will be zero-indexed, so in
            total, would expect inputs from 0 to (NGPUs - 1). This assigns the
            GPU to the particular process.
        start (int): The particle index start - defines which section of the
            external and internal arrays to read from
        end (int): The particle index end - defines which section of the
            external and internal arrays to read from
        external (np.array): External degrees of freedom for ALL particles, not
            just those assigned to this GPU
        internal (np.array): Internal degrees of freedom for ALL particles, not
            just those assigned to this GPU
        structure_files (dict): The filenames and some structure settings needed
            to generate a temporary structure object. This is needed as it's not
            possible to pickle the structure object directly.
        minimiser_settings (dict): Settings needed for the local optimisation

    Returns:
        dict: standard gallop results dictionary, as the value of a dict with
            the GPU value as the key. This is used to reconstruct the full
            results needed for the particle swarm update step.
    """
    minimiser_settings["device"] = torch.device("cuda:"+str(GPU))
    external = external[start:end]
    internal = internal[start:end]
    temp_struct = structure.Structure()
    temp_struct.from_json(structure_files, is_json=False)
    result = optimiser.minimise(temp_struct, external=external, internal=internal,
                                **minimiser_settings)
    result = {GPU : result}
    return result


def minimise(iteration, struct, swarm, external, internal, GPU_split,
            minimiser_settings, start_time=None):
    """[summary]

    Args:
        iteration (int): The current GALLOP iteration
        struct (gallop.Structure): GALLOP Structure object
        swarm (gallop.Swarm): GALLOP Swarm object
        external (np.array): Numpy array of the external degrees of freedom
        internal (np.array): Numpy array of the internal degrees of freedom
        GPU_split (List of lists): List of lists with the following structure:
                    [[GPU_1, % on GPU_1],
                    [GPU_2, % on GPU_2],
                                ... ...
                    [GPU_N, % on GPU_N]]
            The GPU IDs are integers that correspond to the index obtained using
            torch.cuda.device_count() and torch.cuda.get_device_name(i) where
            i is produced by range(torch.cuda.device_count()). The percentage is
            expressed in the range [0,100] rather than [0,1].
        minimiser_settings (dict): Dictionary with the settings needed for the
            local optimiser
        start_time (float, optional): The time at which the runs started. Used
            in the save_CIF function to add timestamps to the files.
            Defaults to None.

    Returns:
        dict: Standard gallop results dictionary as would normally be obtained
        using the optimiser.minimise function
    """
    structure_files = struct.to_json(return_json=False)
    minimiser_settings["streamlit"] = False
    minimiser_settings["save_CIF"] = False
    common_args = [external, internal, structure_files, minimiser_settings]
    args = []
    devices = []
    for i, g in enumerate(GPU_split):
        gpuid = int(g[0])
        devices.append(gpuid)
        percentage = g[1] / 100.
        if i == 0:
            start = 0
            end = int(np.ceil(percentage*swarm.n_particles))
        else:
            start = end
            end = start + int(np.ceil(percentage*swarm.n_particles))
        args.append([gpuid,start,end]+common_args)
    if start_time is None:
        start_time = time.time()
    with mp.Pool(processes = len(GPU_split)) as p:
        results = p.starmap(multiGPU, args)
    p.close()
    p.join()
    combined = results[0]
    for x in results[1:]:
        combined.update(x)
    # Now reconstruct the full results dict
    result = {"GALLOP Iter" : iteration}
    for d in devices:
        if "external" in result.keys():
            result["external"] = np.vstack([result["external"],
                                            combined[d]["external"]])
        else:
            result["external"] = combined[d]["external"]
        if "internal" in result.keys():
            result["internal"] = np.vstack([result["internal"],
                                            combined[d]["internal"]])
        else:
            result["internal"] = combined[d]["internal"]
        if "chi_2" in result.keys():
            result["chi_2"] = np.hstack([result["chi_2"], combined[d]["chi_2"]])
        else:
            result["chi_2"] = combined[d]["chi_2"]
    files.save_CIF_of_best_result(struct, result, start_time,
                                    minimiser_settings["n_reflections"])
    return result
