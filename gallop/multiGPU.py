import gallop.structure as structure
import gallop.optimiser as optimiser
import torch
import torch.multiprocessing as mp
import numpy as np

def multiGPU(GPU, start, end, external, internal, structure_files, minimiser_settings):
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
        dict: standard gallop results dictionary, with an extra "GPU" field
            which can be used to reconstruct the full results needed for the
            particle swarm update step.
    """
    minimiser_settings["device"] = torch.device("cuda:"+str(GPU))
    external = external[start:end]
    internal = internal[start:end]
    struct = structure.Structure()
            #name=structure_files["name"],
            #ignore_H_atoms=structure_files["ignore_H_atoms"],
            #absorb_H_occu_increase=structure_files["absorb_H_occu_increase"])
    struct.from_json(structure_files)
    #struct.add_data(structure_files["data_file"])
    #for zm in structure_files["zmatrices"]:
    #    struct.add_zmatrix(zm)
    result = optimiser.minimise(struct, external=external, internal=internal,
                                **minimiser_settings)

    result = {GPU : result}
    return result


def minimise(i, struct, swarm, external, internal, GPU_split, minimiser_settings):
    #structure_files = {
    #        "name" : struct.name,
    #        "ignore_H_atoms" : struct.ignore_H_atoms,
    #        "absorb_H_occu_increase" : struct.absorb_H_occu_increase,
    #        "data_file" : struct.data_file,
    #        "source" : struct.source,
    #        "zmatrices" : [z.filename for z in struct.zmatrices]}
    structure_files = struct.to_json()
    minimiser_settings["streamlit"] = False
    common_args = [external, internal, structure_files, minimiser_settings]
    args = []
    devices = []
    for i, g in enumerate(GPU_split):
        id = int(g[0])
        devices.append(id)
        percentage = g[1] / 100.
        if i == 0:
            start = 0
            end = int(np.ceil(percentage*swarm.n_particles))
        else:
            start = end
            end = start + int(np.ceil(percentage*swarm.n_particles))
        args.append([id,start,end]+common_args)
    args[0][-1]["streamlit"] = True

    with mp.Pool(processes = len(GPU_split)) as p:
        results = p.starmap(multiGPU, args)
    p.close()
    p.join()
    combined = results[0]
    [combined.update(x) for x in results[1:]]
    result = {"GALLOP Iter" : i}
    for d in devices:
        if "external" in result.keys():
            result["external"] = np.vstack([result["external"], combined[d]["external"]])
        else:
            result["external"] = combined[d]["external"]
        if "internal" in result.keys():
            result["internal"] = np.vstack([result["internal"], combined[d]["internal"]])
        else:
            result["internal"] = combined[d]["internal"]
        if "chi_2" in result.keys():
            result["chi_2"] = np.hstack([result["chi_2"], combined[d]["chi_2"]])
        else:
            result["chi_2"] = combined[d]["chi_2"]

    return result