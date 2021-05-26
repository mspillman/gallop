# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for local optimisation and a class for particle swarm.
"""

import time
import os
import random
import torch
import tqdm
import pyDOE
import numpy as np
import matplotlib.pyplot as plt
import torch_optimizer as t_optim
from scipy.stats import gaussian_kde

from gallop import chi2
from gallop import tensor_prep
from gallop import files
from gallop import intensities
from gallop import zm_to_cart


def seed_everything(seed=1234, change_backend=False):
    """
    Set random seeds for everything.
    Note that at the moment, CUDA (which is used by PyTorch) is not
    deterministic for some operations and as a result, GALLOP runs from
    the same seed may still produce different result.
    See here for more details:
        https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        seed (int, optional): Set the random seed to be used.
            Defaults to 1234.
        change_backend (bool, optional): Whether to change the backend used to
            try to make the code more reproducible. At the moment, it doesn't
            seem to help... default to False
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if change_backend:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_minimiser_settings(Structure):
    """
    Get a dictionary of the settings used for the minimise function
    so it can be easily modified and passed to the function.

    Args:
        Structure (class): GALLOP structure

    Returns:
        dict: Dictionary of the settings for the minimise function
            Keys = n_reflections, include_dw_factors, chi2_solved, n_iterations,
            n_cooldown, learning_rate, learning_rate_schedule, verbose,
            use_progress_bar, print_every, check_min, dtype, device,
            ignore_reflections_for_chi2_calc, optimizer, loss, eps, save_CIF,
            streamlit, torsion_shadowing, Z_prime, use_restraints
    """
    settings = {}
    settings["n_reflections"] = len(Structure.hkl)
    settings["include_dw_factors"] = True
    settings["chi2_solved"] = None
    settings["n_iterations"] = 500
    settings["n_cooldown"]   = 100
    settings["learning_rate"] = 5e-2
    settings["learning_rate_schedule"] = "1cycle"
    settings["verbose"] = False
    settings["use_progress_bar"] = True
    settings["print_every"] = 100
    settings["check_min"] = 100
    settings["dtype"] = torch.float32
    settings["device"] = None
    settings["ignore_reflections_for_chi2_calc"] = False
    settings["optimizer"] = "adam"
    settings["loss"] = "xlogx"
    settings["eps"] = 1e-8
    settings["save_CIF"] = True
    settings["streamlit"] = False
    settings["torsion_shadowing"] = False
    settings["Z_prime"] = 1
    settings["use_restraints"] = False
    return settings


def adjust_lr_1_over_sqrt(optimizer, iteration, learning_rate):
    """
    Decay the learning_rate used in the local optimizer by 1/sqrt(iteration+1)
    Called by the minimise function

    Args:
        optimizer (pytorch optimizer): A pytorch compatible optimizer
        iteration (int): The current iteration
        learning_rate (float): The initial learning rate used

    Returns:
        float: the new learning rate
    """
    lr = learning_rate * 1/np.sqrt((iteration)+1.)
    #lr = learning_rate * 1/np.sqrt((iteration/2.)+1.)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_lr_1_cycle(optimizer, iteration, low, high, final, num_iterations,
                        upperb1=0.95, lowb1=0.85, upperb2=0.9, lowb2=0.9):
    """
    See here: https://sgugger.github.io/the-1cycle-policy.html

    Args:
        optimizer (pytorch optimizer): the optimizer object
        iteration (int): current learning rate
        low (float): initial learning rate
        high (float): maximum learning rate
        final (float): the cooldown iterations at the end of the run
        num_iterations (int): the number of iterations that will be performed
        upperb1 (float, optional): maximum beta1 / momentum. Defaults to 0.95.
        lowb1 (float, optional): minimum beta1 / momentum. Defaults to 0.85.
        upperb2 (float, optional): maximum beta2. Defaults to 0.9.
        lowb2 (float, optional): minimum beta2. Defaults to 0.9.

    Returns:
        float: the learning rate
    """
    increment = (high-low)/(final//2)
    b1_increment = (upperb1 - lowb1) / (final//2)
    b2_increment = (upperb2 - lowb2) / (final//2)
    final_increment = low/(num_iterations-final)
    lr, b1, b2 = 0, 0, 0
    if iteration < final//2:
        lr = increment*(iteration)+low
        b1 = upperb1 - b1_increment*iteration
        b2 = lowb2 + b2_increment*iteration
    elif iteration <= final:
        lr = ((increment*(final//2))+low) - (increment*((iteration)-(final//2)))
        b1 = lowb1 + b1_increment*(iteration-(final//2))
        b2 = upperb2 - b2_increment*(iteration-(final//2))
    else:
        lr = low - (final_increment*(iteration-final))
        b1 = upperb1
        b2 = lowb2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = [b1, b2]
        param_group["momentum"] = b1
    return lr

def find_learning_rate(Structure, external=None, internal=None,
    min_lr=-4, max_lr=np.log10(0.15), n_trials=200, minimiser_settings=None,
    plot=False, multiplication_factor=0.75, logplot=True, figsize=(10,6)):
    """
    See here: https://sgugger.github.io/the-1cycle-policy.html
    In contrast to the advice in the article (which is for training neural
    networks), in this work, it seems to work better to take the absolute
    minimum point that is obtained.

    Args:
        min_lr (int, optional): Minimum (log10) learning rate to try.
            Defaults to -5 (which is equivalent to 10^-4).
        max_lr (int, optional): Maximum (log10) learning rate to try.
            Defaults to log10(0.15).
        n_trials (int, optional): Number of trials between min_lr and max_lr.
            Defaults to 200.
        multiplication_factor (float, optional): multiply the lowest points
            in the learning rate curve by a fixed amount, e.g. 0.5 or 2.0 to
            decrease and increase the learning rate respectively. Defaults to 1.
        plot (bool, optional): Plot the curve of learning rate vs loss.
            Defaults to True.
        logplot (bool, optional): If plotting, use a logarithmic x-axis.
            Defaults to True.
        figsize (tuple, optional): The size of the figure. Defaults to (10,6)

    Returns:
        tuple: Tuple containing the trial learning rate values, the losses
                obtained, the multiplication_factor and the learning rate
                associated with the minimum loss value multiplied by the
                multiplication_factor
    """

    if minimiser_settings is not None:
        # Set the learning rates to be tested
        trial_values = np.logspace(min_lr, max_lr, n_trials)
        lr_minimiser_settings = minimiser_settings.copy()
        lr_minimiser_settings["learning_rate"] = trial_values
        lr_minimiser_settings["learning_rate_schedule"] = "array"
        lr_minimiser_settings["n_iterations"] = len(trial_values)
        lr_minimiser_settings["run"] = -1
        lr_minimiser_settings["external"] = external
        lr_minimiser_settings["internal"] = internal
        lr_minimiser_settings["Structure"] = Structure
        lr_minimiser_settings["save_CIF"] = False
        lr_minimiser_settings["save_loss"] = True


        # Get the losses at each learning rate
        result = minimise(**lr_minimiser_settings)

        losses = result["losses"]


        if plot:
            plt.figure(figsize=figsize)
            plt.plot(trial_values, losses)
            if logplot:
                plt.xscale('log')
            plt.show()
        if multiplication_factor is None:
            minpoint = np.argmin(losses)
            final_1 = (trial_values[minpoint:]
                        - trial_values[minpoint:].min())[-1]
            final_0pt5 = 0.5 * final_1
            final_0pt25 = 0.25 * final_1
            if losses[-1] < final_0pt25:
                multiplication_factor = 1.0
            elif losses[-1] < final_0pt5:
                multiplication_factor = 0.75
            else:
                multiplication_factor = 0.5

        minimum_point = trial_values[losses == losses.min()][0]
        return trial_values, losses, multiplication_factor, multiplication_factor * minimum_point
    else:
        print("ERROR! Minimiser Settings are needed!")
        return None


def minimise(Structure, external=None, internal=None, n_samples=10000,
    n_iterations=500, n_cooldown=100, device=None, dtype=torch.float32,
    n_reflections=None, learning_rate_schedule="1cycle",
    b_bounds_1_cycle=None, check_min=1, optimizer="Adam", verbose=False,
    print_every=100, learning_rate=3e-2, betas=None, eps=1e-8, loss="sum",
    start_time=None, run=1, torsion_shadowing=False, Z_prime=1,
    save_trajectories=False, save_grad=False, save_loss=False,
    include_dw_factors=True, chi2_solved=None,
    ignore_reflections_for_chi2_calc=False, use_progress_bar=True,
    save_CIF=True, streamlit=False, use_restraints=False, include_PO=False,
    PO_axis=(0, 0, 1)):
    """
    Main minimiser function used by GALLOP. Take a set of input external and
    internal degrees of freedom (as numpy arrays) together with the observed
    intensities and inverse covariance matrix, and optimise the chi-squared
    factor of agreement between the calculated and observed intensities.

    This forms one part of the GALLOP algorithm, the other part is a particle
    swarm optimisation step which is used to generate the starting positions
    that are passed to this minimisation function.

    Args:
        Structure (Structure object):   Contains all of the information about
            the crystal structure including PXRD data, Z-matrices, unit cell etc
        external (Numpy array, optional):   The external degrees of freedom for
            n_samples. Defaults to None. If None, then these will be randomly
            generated.
        internal (Numpy array, optional):   The internal degrees of freedom for
            n_samples. Defaults to None. If None, then these will be randomly
            generated.
        n_samples (int, optional):  If the external/internal DoFs are None,
            how many samples to generate. Defaults to 10000.
        n_iterations (int, optional): Total number of iterations to run the
            local-optimisation algorithm. Defaults to 500.
        n_cooldown (int, optional): Used in the 1-cycle learning rate policy.
            The number of iterations at the end of a local optimisation run with
            a low learning rate. Defaults to 100.
        device (torch.device, optional):    Where to run the calculations. If
            None, then check to see if a GPU exists and use it by default.
        dtype (torch datatype, optional): What datatype to use. Defaults to
            torch.float32.
        n_reflections (int, optional): The number of reflections to use for the
            chi_2 calculation. If None, use all available.
        learning_rate_schedule (str, optional): How to modify the learning rate
            during the optimisation process.
            One of: "1cycle", "sqrt", "constant", "array".
            "1cycle"    -   rapidly increase then decrease the learning rate,
                            before a "cooldown" period with a lower learning
                            rates. See this link for more details:
                            https://sgugger.github.io/the-1cycle-policy.html
            "sqrt"      -   decay the learning rate after each iteration
                            according to:
                                lr = learning_rate * 1/np.sqrt((iteration)+1.)
            "constant"  -   constant learning rate
            "array"     -   read the learning rate at each iteration from an
                            array
            Defaults to "1cycle".
        learning_rate (Float or array, optional):   Depends on
            learning_rate_schedule. If float, then this is the initial learning
            rate used with the decay schedule. If an array, then the learing
            rate will be read from the array at each iteration.
            Defaults to 3e-2.
        b_bounds_1_cycle (dict, optional): Used in the 1cycle learning rate
            policy to set the upper and lower beta values. If set to None,
            then the following values will be used:
            {"upperb1":0.95, "lowb1":0.85, "upperb2":0.9, "lowb2":0.9}
        check_min (int, optional):  The number of iterations after which the
            best value of chi2 obtained so far is checked. Defaults to 1.
        optimizer (str or torch-compatible optimizer): Either a string or a
            torch-compatible optimizer. The only strings currently used are
            "Adam", "DiffGrad" and "Yogi".
        verbose (bool, optional): Print out information during the run.
            Defaults to False.
        print_every (int, optional):    If verbose is True, then how frequently
            to print out the information on the run progress. Also controls
            how often the best chi2 value is updated for the progress-bar.
        betas (list, optional): beta1 and beta2 values to use in Adam. Defaults
            to [0.9, 0.9] (if None set) which is equivalent to
            [beta1=0.9, beta2=0.9]
        eps (float, optional): Epsilon value to use in Adam or Adam derived
            optimizers. Defaults to 1e-8.
        loss (str or function, optional):   Pytorch requires a scalar to
            determine the gradients so the series of chi2 values obtained from
            the n_samples must be combined through some function before the
            gradient is determined. Different functions can be used to scaled
            the gradients in different ways. This may be advantageous if there
            is a significant difference in the magnitude of the gradient when
            chi2 is large vs when it is small. If string, must be one of:
            sum, sse, xlogx
                sum     -   add all the chi2 values together. The gradient for
                            each independent run will be the derivative of chi2
                            with respect to the degrees of freedom.
                sse     -   sum(chi2^2) (sum of squared errors). The gradient
                            for each independent run will be the derivative of
                            chi2 (wrt DoF) multiplied by 2*chi2.
                xlogx   -   sum(chi2 * log(chi2)). The gradient for each
                            independent run will be the derivative of chi2 (wrt
                            DoF) multiplied by log(chi2) + 1.
            If a function, this function must must take as input a pytorch
            tensor and return a scalar.
            Defaults to "sum".
        start_time (time.time(), optional): The start time of a run or set of
            runs. Useful in full GALLOP runs, but if set to None then the start
            time will automatically be determined.
        run (int, optional): If a set of runs are being performed, then the run
            number can be passed for printing. Defaults to 1.
        torsion_shadowing (bool, optional) : Pin the torsion angles for Z'>1
            structures so that all fragments have the same torsion angles. This
            can result in faster solutions, and once a "good" result is found,
            this can be set to False to allow each fragment to refine freely.
            Defaults to False
        Z_prime (int, optional): If using torsion_shadowing, this is the Z' for
            the current unit cell. Defaults to 1.
        save_trajectories (bool, optional): Store the DoF, chi_2 and loss value
            after every iteration. This will be slow, as it requires transfer
            from the GPU to CPU. Defaults to False.
        save_grad (bool, optional): Store the gradients of the internal and
            external DoF with respect to the chi_2 / loss values after every
            iteration. This will be slow, as it requires transfer from the GPU
            to CPU. Defaults to False.
        save_loss (bool, optional) : Save the value of the loss at each
            iteration. Used by find_learning_rate. Defaults to False.
        include_dw_factors (bool, optional): Include Debye-Waller factors in
            the intensity calculations. Defaults to True.
        chi2_solved (float, optional):  The value below which a structure is
            considered solved (if known). Used for some printed information.
            If None, then this is ignored. Defaults to None.
        ignore_reflections_for_chi2_calc (bool, optional):  The normal chi2
            calculation divides the output by (n_reflections - 2)
            i.e. chi2 = d.A.d / (n_reflections - 2)
            If set to True, this reverses this operation. Defaults to False.
        use_progress_bar (bool, optional): Use a progress bar to provide visual
            feedback on run progress. Defaults to True.
        save_CIF (bool, optional): Save a CIF of the best structure found
            after optimising. Defaults to True.
        streamlit (bool, optional): If using the streamlit webapp interface,
            this is set to True to enable a progress bar etc. Defaults to False
        use_restraints (bool, optional): Use distance-based restraints as an
            additional penalty term in the loss function. Must be added to the
            Structure object via Structure.add_restraint() before use. Defaults
            to False.
        include_PO (bool, optional): Include a preferred orientation correction
            to the intensities during optimization. This is a global parameter
            applied to all independent local optimisation runs. Defaults to
            False.
        PO_axis (tuple, optional): The axis along which to apply the
            March-Dollase PO correction. Defaults to (0, 0, 1).

    Returns:
        dictionary: A dictionary containing the optimised external and internal
            degrees of freedom and their associated chi_2 values.
            If save_trajectories is True then this also contains the
            trajectories of the particles, the chi_2 values and loss values
            at each iteration.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.float32
    if b_bounds_1_cycle is None:
        b_bounds_1_cycle = {"upperb1":0.95, "lowb1":0.85, "upperb2":0.9,
                            "lowb2":0.9}
    if betas is None:
        betas = [0.9,0.9]
    if streamlit:
        import streamlit as st

    # Load the tensors and other parameters needed
    tensors = tensor_prep.get_all_required_tensors(
                                Structure, external=external, internal=internal,
                                n_samples=n_samples, device=device, dtype=dtype,
                                n_reflections=n_reflections, verbose=verbose,
                                include_dw_factors=include_dw_factors)
    trajectories = []
    gradients = []
    losses = []
    if torsion_shadowing:
        # Only pay attention to the first block of torsions accounting for Z'>1.
        # This assumes that all of the fragments have similar torsion angles.
        # This can significantly speed up solving Z'>1 structures where this
        # assumption is valid.
        # To use this, the ZMs supplied must be in blocks that correspond to the
        # unique fragments in Z', for example, a structure with two flexible
        # fragments and two ions would need to be entered into the structure as
        # ion1 fragment1 ion2 fragment2
        # This is different to the way DASH converts ZMs by default, where ions
        # tend to come as zm1 and zm2, then the flex fragments as zm3 and zm4.
        n_torsion_fragments = len(tensors["zm"]["torsion"])
        first_frag = int(n_torsion_fragments / Z_prime)
        tensors["zm"]["torsion"] = tensors["zm"]["torsion"][:first_frag]*Z_prime

    if use_restraints:
        if Structure.ignore_H_atoms:
            restraints = np.array(Structure.restraints_no_H)
        else:
            restraints = np.array(Structure.restraints)
        atoms = restraints[:,:2]
        distances = restraints[:,2]
        percentages = restraints[:,3] / 100.

        atoms = torch.from_numpy(atoms).type(torch.long).to(device)
        distances = torch.from_numpy(distances).type(dtype).to(device)
        percentages = torch.from_numpy(percentages).type(dtype).to(device)
        lattice_matrix = torch.from_numpy(
                    np.copy(Structure.lattice.matrix)).type(dtype).to(device)


    # Initialize the optimizer
    if isinstance(optimizer, str):
        if learning_rate_schedule.lower() == "array":
            init_lr = learning_rate[0]
        else:
            init_lr = learning_rate
        if optimizer.lower() == "adam":
            optimizer = torch.optim.Adam([tensors["zm"]["external"],
                                        tensors["zm"]["internal"]],
                                        lr=init_lr, betas=betas, eps=eps)
        elif optimizer.lower() == "yogi":
            optimizer = t_optim.Yogi([tensors["zm"]["external"],
                                        tensors["zm"]["internal"]],
                                        lr=init_lr, betas=betas, eps=eps)
        elif optimizer.lower() == "diffgrad":
            optimizer = t_optim.DiffGrad([tensors["zm"]["external"],
                                        tensors["zm"]["internal"]],
                                        lr=init_lr, betas=betas, eps=eps)
        else:
            print("Only supported optimizers are \"Adam\", \"DiffGrad\" or",
            "\"Yogi\".")
            exit()
    else:
        # Ensure that the optimizer is a torch compatible optimizer that
        # accepts learning rate changes
        if not hasattr(optimizer, "param_groups"):
            print("Optimizer not compatible")
            exit()
        else:
            optimizer.param_groups[0]["params"] = [tensors["zm"]["external"],
                                                    tensors["zm"]["internal"]]
            for param_group in optimizer.param_groups:
                if learning_rate_schedule.lower() == "array":
                    param_group['lr'] = learning_rate[0]
                else:
                    param_group['lr'] = learning_rate

    if include_PO:
        PO_axis = np.array(PO_axis)
        u = Structure.hkl / np.sqrt(np.einsum("kj,kj->k",
                Structure.hkl, np.einsum("ij,kj->ki",
                    Structure.lattice.reciprocal_lattice.matrix,
                    Structure.hkl))).reshape(-1,1)
        cosP = np.einsum("ij,j->i", u, np.inner(
                        Structure.lattice.reciprocal_lattice.matrix, PO_axis))
        one_minus_cosPsqd = 1.0-cosP**2
        one_minus_cosPsqd[one_minus_cosPsqd < 0.] *= 0.
        sinP = np.sqrt(one_minus_cosPsqd)
        cosP = torch.from_numpy(cosP).type(dtype).to(device)
        sinP = torch.from_numpy(sinP).type(dtype).to(device)
        factor = torch.Tensor([1.0]).type(dtype).to(device)
        factor.requires_grad = True
        optimizer.param_groups[0]["params"] += [factor]

    if start_time is None:
        t1 = time.time()
    else:
        t1 = start_time

    # Add the progress bar, if using
    if use_progress_bar and not streamlit:
        iters = tqdm.trange(n_iterations)
    else:
        iters = range(n_iterations)
        if streamlit:
            prog_bar = st.progress(0.0)

    # Now perform the optimisation iterations
    for i in iters:
        # Zero out gradient, else they will accumulate between iterations
        optimizer.zero_grad()

        if learning_rate_schedule.lower() == "1cycle":
            lr = adjust_lr_1_cycle(optimizer, i, learning_rate/10,
                    learning_rate, n_iterations-n_cooldown, n_iterations,
                    **b_bounds_1_cycle)
        elif learning_rate_schedule.lower() == "sqrt":
            lr = adjust_lr_1_over_sqrt(optimizer, i, learning_rate)
        elif learning_rate_schedule.lower() == "constant":
            lr = learning_rate
        elif learning_rate_schedule.lower() == "array":
            try:
                lr = learning_rate[i]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            except IndexError:
                print("Error in learning rate array")
                if len(learning_rate) < i:
                    print("Insufficient entries in list, using last value")
                    lr = learning_rate[-1]
                else:
                    print("Check and try again")
                    exit()
        else:
            if i == 0:
                print("Learning rate scheduler unknown, using 1cycle")
                learning_rate_schedule="1cycle"
                lr = adjust_lr_1_cycle(optimizer, i, learning_rate/10,
                        learning_rate, n_iterations-n_cooldown, n_iterations)
            else:
                print("An error has occurred with lr scheduling")

        # Forward pass - this gets a tensor of shape (n_samples, 1) with a
        # chi_2 value for each set of external/internal DoFs.
        if use_restraints or include_PO:
            asymmetric_frac_coords = zm_to_cart.get_asymmetric_coords(
                                                                **tensors["zm"])

            calculated_intensities = intensities.calculate_intensities(
                            asymmetric_frac_coords, **tensors["int_tensors"])
            if include_PO:
                corrected_intensities = intensities.apply_MD_PO_correction(
                                            calculated_intensities,
                                            cosP, sinP, factor)
                chi_2 = chi2.calc_chisqd(corrected_intensities,
                                        **tensors["chisqd_tensors"])
            else:
                chi_2 = chi2.calc_chisqd(calculated_intensities,
                                        **tensors["chisqd_tensors"])
        else:
            chi_2 = chi2.get_chi_2(**tensors)
            if ignore_reflections_for_chi2_calc:
                # Counteract the division normally used in chi2 calculation
                chi_2 *= (tensors["intensity"]["hkl"].shape[1] - 2)

        if use_restraints:
            with torch.no_grad():
                min_chi_2 = chi_2.min()
            cart = torch.einsum("jk,bilj->bilk",
                            lattice_matrix,asymmetric_frac_coords[:,atoms,:])
            atomic_distances = torch.sqrt(
                            ((cart[:,:,0,:] - cart[:,:,1,:])**2).sum(dim=-1))
            distance_penalty = (min_chi_2*percentages*(
                                            distances-atomic_distances)**2).sum()
        else:
            distance_penalty = 0
        # PyTorch expects a single value for backwards pass.
        # Need a function to convert all of the chi_2 values into a scalar
        if isinstance(loss, str):
            if loss.lower() == "sse":
                L = (chi_2**2).sum() + distance_penalty
            elif loss.lower() == "sum":
                L = chi_2.sum() + distance_penalty
            elif loss.lower() == "xlogx":
                L = torch.sum(chi_2*torch.log(chi_2)) + distance_penalty
        else:
            if loss is None:
                # Default to the sum operation if loss is None
                L = chi_2.sum() + distance_penalty
            else:
                try:
                    L = loss(chi_2, distance_penalty)
                except TypeError:
                    try:
                        L = loss(chi_2)
                    except RuntimeError:
                        print("Unknown / incompatible loss function",loss)

        # Backward pass to calculate gradients
        L.backward()
        if save_loss:
            losses.append(L.detach().cpu().numpy())
        if i == 0:
            best = torch.min(chi_2).item()
            best_iteration = 1
        else:
            if i % check_min == 0:
                if torch.min(chi_2).item() < best:
                    best = torch.min(chi_2).item()
                    best_iteration = i
        i+=1
        if verbose:
            if i % print_every == 0 or i == 1:
                detached_chi2 = chi_2.detach().cpu().numpy()
                if chi2_solved is not None:
                    n_solved = detached_chi2[detached_chi2 < chi2_solved]
                    printstring = (
                        "GALLOP iter {:04d} | LO iter {:04d} | lr {:.3f} ||",
                        "max/mean/min chi^2 {:.1f} / {:.1f} / {:.1f} ||",
                        "Time {:.1f} (s) / {:.1f} (min) || Best {:.1f}",
                        "in iter {:04d} || n<{:.1f}: {:05d}")
                    print("".join(printstring).format(
                            run+1, i, lr,
                            chi_2.max().item(), chi_2.mean().item(),
                            chi_2.min().item(), time.time() - t1,
                            (time.time() - t1)/60, best, best_iteration,
                            chi2_solved, n_solved.shape[0]))
                else:
                    printstring = (
                        "GALLOP iter {:04d} | LO iter {:04d} | lr {:.3f} ||",
                        "max/mean/min chi^2 {:.1f} / {:.1f} / {:.1f} ||",
                        "Time {:.1f} (s) / {:.1f} (min) || Best {:.1f}"
                        "in iter {:04d}")
                    print("".join(printstring).format(
                            run+1, i, lr,
                            chi_2.max().item(), chi_2.mean().item(),
                            chi_2.min().item(), time.time() - t1,
                            (time.time() - t1)/60, best,
                            best_iteration))
        elif use_progress_bar and not streamlit:
            if i % print_every == 0 or i == 1:
                iters.set_description(
                    "GALLOP iter {:04d} LO iter {:04d} min chi2 {:.1f}".format(
                        run+1, i, chi_2.min().item()))
        if save_trajectories:
            trajectories.append(
                    [tensors["zm"]["external"].detach().cpu().numpy(),
                    tensors["zm"]["internal"].detach().cpu().numpy(),
                    chi_2.detach().cpu().numpy(),
                    L.detach().cpu().numpy()])
        if save_grad:
            gradients.append(
                [tensors["zm"]["external"].grad.detach().cpu().numpy(),
                tensors["zm"]["internal"].grad.detach().cpu().numpy()])
        if i != n_iterations:
            optimizer.step()
        if streamlit:
            prog_bar.progress(i/n_iterations)
    result = {
            "external"     : tensors["zm"]["external"].detach().cpu().numpy(),
            "internal"     : tensors["zm"]["internal"].detach().cpu().numpy(),
            "chi_2"        : chi_2.detach().cpu().numpy(),
            "GALLOP Iter"  : run
            }
    if torsion_shadowing:
        torsions = result["internal"]
        torsions = torsions[:,:int(torsions.shape[1] / Z_prime)]
        torsions = np.tile(torsions, (1,Z_prime))
        result["internal"] = torsions
    if save_CIF:
        files.save_CIF_of_best_result(Structure, result, start_time,
                                        n_reflections)
    if save_trajectories:
        result["trajectories"] = trajectories
    if save_loss:
        result["losses"] = np.array(losses)
    if save_grad:
        result["gradients"] = gradients
    del tensors

    return result

def plot_torsion_difference(Structure, result, n_swarms=1,
    verbose=False, figsize=(10,10), xlim=None,
    ylim=None, cmap="tab20", call_show=True):
    """
    Calculate the average difference between the known torsions (obtained
    from the Z-matrix input) to those obtained in an SDPD attempt.

    Torsion angles are first cast into the plane in order to account for the
    fact that 0 deg == 360 deg.

    Args:
        Structure (class): Structure object containing the true torsions,
            read in from the Z-matrices.
        result (dict): A result dict returned by the minimise function, which
            contains the internal DoF and the chi_2 values.
        n_swarms (int, optional): plot the separate swarms in different colours
            or set to 1 to plot all with the same colour. Defaults to 1.
        verbose (bool, optional): Print out information. Defaults to False.
        figsize (tuple, optional): Size of the plot. Defaults to (10,10).
        xlim (dict, optional): Limits of the x-axis.
            Defaults to {"left" : 0, "right" : None}.
        ylim (dict, optional): Limits of the y-axis.
            Defaults to {"bottom" : 0, "top" : None}.
        cmap (str, optional): the matplotlib colourmap to use.
            Defaults to tab20
        call_show (bool, optional): If True, will call plt.show() to render the
            plot. If False, the plot can be used in a subplot along with other
            plots by the user in the parent script or notebook.
    """
    if xlim is None:
        xlim = {"left" : 0, "right" : None}
    if ylim is None:
        ylim = {"bottom" : 0, "top" : None}
    true_torsions = Structure.zm_torsions
    internal = result["internal"]
    chi_2 = result["chi_2"]
    subswarm = internal.shape[0] // n_swarms
    diff = true_torsions - internal
    sindiff = np.sin(diff)
    cosdiff = np.cos(diff)

    if verbose:
        if chi_2 is not None:
            mean_ang_diff = np.arctan2(sindiff.mean(axis=1),
                                        cosdiff.mean(axis=1))
            mean_ang_diff = (180/np.pi)*(mean_ang_diff)
            print("Mean diff, min mean diff, chi2 min mean diff")
            print(np.abs(mean_ang_diff).mean(), np.abs(mean_ang_diff).min(),
                                            mean_ang_diff[chi_2 == chi_2.min()])

    c_coord_diff = np.cos(true_torsions) - np.cos(internal)
    s_coord_diff = np.sin(true_torsions) - np.sin(internal)
    dist = np.sqrt((c_coord_diff**2 + s_coord_diff**2).mean(axis=1))
    if call_show:
        plt.figure(figsize=figsize)
    plt.scatter(chi_2, dist, s=10, cmap=cmap, alpha=0.75,
                    c=(np.arange(internal.shape[0])//subswarm))
    plt.xlim(**xlim)
    plt.ylim(**ylim)
    if call_show:
        plt.show()


class Swarm(object):
    def __init__(self, Structure, n_particles=10000, n_swarms=10,
        particle_best_position = None, best_chi_2 = None, velocity = None,
        position = None, best_subswarm_chi2 = None, inertia="ranked", c1=1.5,
        c2=1.5, inertia_bounds=(0.4,0.9), use_matrix=True, limit_velocity=True,
        global_update=False, global_update_freq=10, vmax=1):
        """
        Class for the particle swarm optimiser used in GALLOP.

        Args:
            Structure (class): GALLOP structure object
            n_particles (int, optional): The number of particles to optimise.
                Defaults to 10000.
            n_swarms (int, optional): The number of independent swarms which
                are represented by the n_particles. Defaults to 20.
            particle_best_position (numpy array, optional): the best position
                on the hypersurface obtained by each particle. Defaults to None.
            best_chi_2 (numpy array, optional): The best chi_2 obtained by each
                particle. Defaults to None.
            velocity (numpy array, optional): The current velocity of the
                particles. Defaults to None.
            position (numpy array, optional): The current position of the
                particles. Defaults to None.
            best_subswarm_chi2 (list, optional): The best chi_2 found in each
                subswarm. Defaults to [].
            inertia (float or str, optional): The inertia to use in the velocity
                update. If random, sample the inertia from a uniform
                distribution. If "ranked", then solutions ranked in order of
                increasing chi2. Lowest chi2 assigned lowest inertia, as defined
                by bounds in inertia_bounds. Defaults to "ranked".
            c1 (int, optional): c1 (social) parameter in PSO equation.
                Defaults to 1.5
            c2 (int, optional): c2 (cognitive) parameter in PSO equation.
                Defaults to 1.5
            inertia_bounds (list, optional): The upper and lower bound of the
                values that inertia will take if inertia is set to "random" or
                "ranked".
                Defaults to [0.4,0.9].
            use_matrix (bool, optional): Take a different step size in every
                degree of freedom. Defaults to True.
            limit_velocity (bool, optional): Restrict the velocity to the range
                (-vmax, vmax). Defaults to True.
            global_update (bool, optional): If True, allow global updates (see
                below). Defaults to False.
            global_update_freq (int, optional): If using subswarms, it may be
                desirable to occasionally update all subswarms swarm as a single
                swarm to allow communication of information from different
                regions of the hypersurface. Setting this to an integer will
                activate the global update when:
                    run number % global_update_freq == 0 and run number > 0
                Defaults to 10.
            vmax (float, optional): The absolute maximum velocity a particle can
                achieve if limit_velocity is True.
        """
        self.Structure = Structure
        self.particle_best_position = particle_best_position
        self.best_chi_2 = best_chi_2
        self.velocity = velocity
        self.position = position
        self.n_swarms = n_swarms
        if best_subswarm_chi2 is not None:
            self.best_subswarm_chi2 = best_subswarm_chi2
        else:
            self.best_subswarm_chi2 = []
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.inertia_bounds = inertia_bounds
        self.use_matrix = use_matrix
        self.swarm_progress = []
        self.limit_velocity = limit_velocity
        self.n_particles = n_particles
        self.best_low_res_chi_2 = None
        self.best_high_res_chi_2 = None
        self.global_update = global_update
        self.global_update_freq = global_update_freq
        self.vmax = vmax

    def get_initial_positions(self, method="latin", latin_criterion=None,
                                MDB=None):
        """
        Generate the initial starting points for a GALLOP attempt. The
        recommended method uses latin hypercube sampling which provides a
        more even coverage of the search space than random uniform sampling,
        which can produce clusters or leave regions unexplored.

        Args:
            method (str, optional): The sampling method to use. Can be one of
                "latin" or "uniform". Defaults to "latin".
            latin_criterion (str, optional): The criterion to be used with the
                latin hypercube method. See pyDOE documentation here:
                https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube
                Defaults to None.
            MDB (str, optional): Supply a DASH .dbf containing the Mogul
                Distribution Bias information for the Z-matrices used. They
                must have been entered into DASH in the same order used for
                GALLOP. Defaults to None.

        Returns:
            tuple: Tuple of numpy arrays containing the initial external and
            internal degrees of freedom
        """
        if self.Structure.total_internal_degrees_of_freedom is None:
            self.Structure.get_total_degrees_of_freedom()

        assert method in ["uniform", "latin"], "method must be latin or uniform"
        if self.n_particles % self.n_swarms != 0:
            print("n_particles should be divisible by n_swarms.")
            self.n_particles = self.n_swarms * (self.n_particles//self.n_swarms)
            print("Setting n_particles to", self.n_particles)
        subswarm = self.n_particles // self.n_swarms
        init_external = []
        init_internal = []

        total_pos = self.Structure.total_position_degrees_of_freedom
        total_rot = self.Structure.total_rotation_degrees_of_freedom
        tot_external = total_pos+total_rot
        total_tors = self.Structure.total_internal_degrees_of_freedom
        # Separate hypercube for each subswarm
        for _ in tqdm.tqdm(range(self.n_swarms)):
            if method == "latin":
                all_dof = np.array(pyDOE.lhs(total_pos + total_rot + total_tors,
                            samples=subswarm, criterion=latin_criterion))
                external = all_dof[:,:total_pos+total_rot]
                pos = external[:,:total_pos]
                rot = external[:,total_pos:]
                tor = all_dof[:,total_pos+total_rot:]
                rot -= 0.5
                rot *= 2. # Rotation to range [-1,1]
                tor -= 0.5
                tor *= 2. * np.pi # Torsions to range [-pi,pi]
                init_external.append(np.hstack([pos,rot]))
                init_internal.append(tor)

            else:
                rand_ext = np.random.uniform(-1,1,size=(subswarm,tot_external))
                rand_int = np.random.uniform(-1,1,size=(subswarm,total_tors))
                init_external.append(rand_ext)
                init_internal.append(rand_int)

        init_external = np.vstack(init_external)
        init_internal = np.vstack(init_internal)

        if MDB is not None:
            distributions = []
            with open(MDB) as dbf:
                for line in dbf:
                    line = line.strip().split(" ")
                    if line[1] == "MDB":
                        distributions.append([int(x) for x in line[-19:]])
                    elif line[1] == "LBUB" and line[2] == "-180.00000":
                        distributions.append([10]*19)
            dbf.close()
            distributions = np.array(distributions)
            bins = np.linspace(0, np.pi, distributions.shape[1])
            kdes = []
            for torsion in distributions:
                samples = []
                for i, t in enumerate(torsion):
                    if t > 0:
                        observed = np.linspace(bins[i]-(np.pi/36),
                                                bins[i]+np.pi/36, t)
                        samples.append(np.hstack([observed, -1*observed]))
                kde = gaussian_kde(np.hstack(samples), bw_method=None)
                kdes.append(kde)
            if init_internal.shape[1] != len(kdes):
                print("Not enough MDBs for the number of torsions.")
            else:
                new_internal = []
                for k in kdes:
                    new_internal.append(k.resample(init_internal.shape[0]))
                new_internal = np.vstack(new_internal).T
                init_internal = new_internal
        return init_external, init_internal


    def update_best(self, chi_2):
        """
        Update the swarm with the best position and chi_2 for each particle

        Args:
            chi_2 (numpy array): the most recently obtained chi_2 values
        """
        better = chi_2 < self.best_chi_2
        self.particle_best_position[better] = self.position[better]
        self.best_chi_2[better] = chi_2[better]

    def get_position_from_dof(self, external, internal):
        """
        Particle position values are unbounded, which can cause some issues
        with the swarm updates. This can be remedied in part by normalising
        all of the coordinates into the range -1 to +1.
        It also means that all of the coordinates will have the same range in
        the swarm, allowing easy comparison of exploration directions.

        For torsions, this is simple - merely take sin and cosine of the angles.
        For quaternions, ensuring that they are unit quaternions should do the
        trick.
        For the translations, this function uses the following:
            2 * ((translation % 1) - 0.5)

        Args:
            Structure (class): GALLOP structure which holds information about
                which indices of external and internal correspond to
                translations and rotations
            external (numpy array): External degrees of freedom
            internal (numpy array): Internal degrees of freedom

        Returns:
            numpy array : The normalised and stacked positions of the particles.
                Order is translation, rotation, torsion
        """
        end_of_translations = self.Structure.total_position_degrees_of_freedom
        n_quaternions = self.Structure.total_rotation_degrees_of_freedom // 4
        translation = np.copy(external[:,:end_of_translations])
        translation = translation % 1    # Convert into range(0,1)
        translation *= 2 * np.pi         # Convert into range(0, 2pi)
        translation = np.hstack([np.sin(translation), np.cos(translation)])

        rotation = np.copy(external[:,end_of_translations:])
        rotation_list = []
        for i in range(n_quaternions):
            # Ensure quaternions are unit quaternions
            quaternion = rotation[:,(i*4):(i+1)*4]
            quaternion /= np.sqrt((quaternion**2).sum(axis=1)).reshape(-1,1)
            rotation_list.append(quaternion)
        rotation = np.hstack(rotation_list)
        # Take the sin and cos of the torsions, and stack everything.
        # Range for all parameters is now -1 to +1
        position = np.hstack([translation, rotation,
                                np.sin(internal), np.cos(internal)])

        return position


    def get_new_external_internal(self, position):
        """
        Convert the swarm representation of position back to the external
        and internal degrees of freedom expected by GALLOP

        Args:
            position (numpy array): The internal swarm representation of the
                particle positions, where the positions and torsion angles have
                been projected onto the unit circle.

        Returns:
            tuple: Tuple of numpy arrays containing the external and internal
                degrees of freedom
        """
        total_position = self.Structure.total_position_degrees_of_freedom
        total_rotation = self.Structure.total_rotation_degrees_of_freedom
        total_torsional = self.Structure.total_internal_degrees_of_freedom
        n_quaternions = total_rotation // 4
        end_external = (2*total_position) + total_rotation
        external = np.copy(position[:,:end_external])
        internal = np.copy(position[:,end_external:])
        # Reverse the normalisation of the particle position,
        # back to range 0 - 1
        pos_sines = external[:,:total_position]
        pos_cosines = external[:,total_position:2*total_position]
        # Can now use the inverse tangent to get positions in range -0.5, 0.5
        translations = np.arctan2(pos_sines, pos_cosines) / (2*np.pi)

        rotations = external[:,2*total_position:]
        rotation_list = []
        for i in range(n_quaternions):
            # Ensure the quaternions are unit quaternions
            quaternion = rotations[:,(i*4):(i+1)*4]
            quaternion /= np.sqrt((quaternion**2).sum(axis=1)).reshape(-1,1)
            rotation_list.append(quaternion)
        rotations = np.hstack(rotation_list)

        external = np.hstack([translations, rotations])
        # Revert torsion representation back to angles using the inverse tangent
        internal = np.arctan2(internal[:,:total_torsional],
                            internal[:,total_torsional:])

        return external, internal

    def PSO_velocity_update(self, previous_velocity, position,
        particle_best_pos, best_chi_2, inertia="random", c1=1.5, c2=1.5,
        inertia_bounds=(0.4,0.9), use_matrix=True):
        """
        Update the velocity of the particles in the swarm

        Args:
            previous_velocity (numpy array): Current velocity
            position (numpy array): Current position
            particle_best_pos (numpy array): Best position for each particle
            best_chi_2 (numpy array): Best chi_2 for each particle
            inertia (str, numpy array or float, optional): Inertia to use.
                If string, can currently only be "random" or "ranked".
                If random, then the inertia is randomly set for each particle
                within the bounds supplied in the parameter inertia_bounds.
                If "ranked", then set the inertia values linearly between the
                bounds, with the lowest inertia for the best particle. If a
                float, then all particles are assigned the same inertia.
                Defaults to "random".
            c1 (int, optional): c1 (social) parameter in PSO equation.
                Defaults to 1.5.
            c2 (int, optional): c2 (cognitive) parameter in PSO equation.
                Defaults to 1.5.
            inertia_bounds (tuple, optional): The upper and lower bound of the
                values that inertia can take if inertia is set to "random" or
                "ranked". Defaults to (0.4,0.9)
            use_matrix (bool, optional): Take a different step size in every
                degree of freedom. Defaults to True.

        Returns:
            numpy array: The updated velocity of each particle
        """
        global_best_pos = particle_best_pos[best_chi_2 == best_chi_2.min()]
        if global_best_pos.shape[0] > 1:
            global_best_pos = global_best_pos[0]
        if (not isinstance(inertia, float)
                                    and not isinstance(inertia, np.ndarray)):
            if inertia.lower() == "random":
                inertia = np.random.uniform(inertia_bounds[0],
                                        inertia_bounds[1],
                                        size=(previous_velocity.shape[0], 1))
            elif inertia.lower() == "ranked":
                ranks = np.argsort(best_chi_2) + 1
                inertia = inertia_bounds[0] + (ranks * (inertia_bounds[1]
                                        - inertia_bounds[0]))/ranks.shape[0]
                inertia = inertia.reshape(-1,1)
            elif inertia.lower() == "r_ranked":
                ranks = np.argsort(1/best_chi_2) + 1
                inertia = inertia_bounds[0] + (ranks * (inertia_bounds[1]
                                        - inertia_bounds[0]))/ranks.shape[0]
                inertia = inertia.reshape(-1,1)
            else:
                print("Unknown inertia type!", inertia)
                print("Setting inertia to 0.5")
                inertia = 0.5
        if use_matrix:
            R1 = np.random.uniform(0,1,size=(position.shape[0],
                                            position.shape[1]))
            R2 = np.random.uniform(0,1,size=(position.shape[0],
                                            position.shape[1]))
        else:
            R1 = np.random.uniform(0,1,size=(position.shape[0], 1))
            R2 = np.random.uniform(0,1,size=(position.shape[0], 1))

        new_velocity = (inertia*previous_velocity
                        + c1*R1*(global_best_pos - position)
                        + c2*R2*(particle_best_pos - position))

        return new_velocity

    def get_new_velocities(self, global_update=True, verbose=True):
        """
        Update the particle velocities using the PSO equations.
        Can either update all particles as a single swarm, or treat them as a
        set of independent swarms (or subswarms).

        Args:
            global_update (bool, optional): If True, update all of the particles
                as a single swarm. If False, then update n_swarms separately.
                Defaults to True.
            verbose (bool, optional): Print out if a global update is being
                performed. Defaults to True.
        """
        subswarm = self.n_particles // self.n_swarms
        use_ranked_all = False
        if isinstance(self.inertia, str):
            if self.inertia.lower() == "ranked_all":
                ranks = np.argsort(self.best_chi_2) + 1
                ranked_inertia = (self.inertia_bounds[0]
                            + (ranks * (self.inertia_bounds[1]
                            - self.inertia_bounds[0]))/ranks.shape[0])
                ranked_inertia = ranked_inertia.reshape(-1,1)
                use_ranked_all = True
        if global_update:
            if verbose:
                print("Global")
            if use_ranked_all:
                self.velocity = self.PSO_velocity_update(self.velocity,
                                    self.position, self.particle_best_position,
                                    self.best_chi_2, inertia=ranked_inertia,
                                    c1=self.c1, c2=self.c2,
                                    inertia_bounds=self.inertia_bounds,
                                    use_matrix=self.use_matrix)
            else:
                self.velocity = self.PSO_velocity_update(self.velocity,
                                    self.position, self.particle_best_position,
                                    self.best_chi_2, inertia=self.inertia,
                                    c1=self.c1, c2=self.c2,
                                    inertia_bounds=self.inertia_bounds,
                                    use_matrix=self.use_matrix)
            for j in range(self.n_swarms):
                begin = j*subswarm
                end = (j+1)*subswarm
                self.best_subswarm_chi2.append(self.best_chi_2[begin:end].min())
            self.swarm_progress.append(self.best_subswarm_chi2)
        else:
            subswarm_best = []
            for j in range(self.n_swarms):
                begin = j*subswarm
                end = (j+1)*subswarm
                swarm_v = self.velocity[begin:end]
                swarm_pos = self.position[begin:end]
                swarm_best_pos = self.particle_best_position[begin:end]
                swarm_chi2 = self.best_chi_2[begin:end]
                if use_ranked_all:
                    swarm_ranked_inertia = ranked_inertia[begin:end]
                    new_vel = self.PSO_velocity_update(swarm_v, swarm_pos,
                            swarm_best_pos, swarm_chi2,
                            inertia=swarm_ranked_inertia,
                            c1=self.c1, c2=self.c2,
                            inertia_bounds=self.inertia_bounds,
                            use_matrix=self.use_matrix)
                else:
                    new_vel = self.PSO_velocity_update(swarm_v, swarm_pos,
                            swarm_best_pos, swarm_chi2, inertia=self.inertia,
                            c1=self.c1, c2=self.c2,
                            inertia_bounds=self.inertia_bounds,
                            use_matrix=self.use_matrix)
                self.velocity[begin:end] = new_vel
                subswarm_best.append(swarm_chi2.min())
            self.best_subswarm_chi2 = subswarm_best
            self.swarm_progress.append(self.best_subswarm_chi2)

        if self.limit_velocity:
            unlimited = self.velocity
            self.velocity[unlimited > self.vmax] = self.vmax
            self.velocity[unlimited < -1*self.vmax] = -1*self.vmax

    def update_position(self, result=None, external=None, internal=None,
        chi_2=None, run=None, global_update=False, verbose=True, n_swarms=None):
        """
        Take a set of results from the minimisation algorithm and use
        them to generate a new set of starting points to be minimised. This
        will also update the internal swarm representation of position and
        velocity.

        Args:
            result (dict, optional): The result dict from a GALLOP minimise run.
                Defaults to None.
            external (numpy array, optional): If no result dict is supplied,
                then pass a numpy array of the external DoF. Defaults to None.
            internal (numpy array, optional): If no result dict is supplied,
                then pass a numpy array of the internal DoF. Defaults to None.
            chi_2 (numpy array, optional): If no result dict is supplied,
                then pass a numpy array of the chi_2 values. Defaults to None.
            run (int, optional): If no result dict is supplied, then pass the
                run number. Defaults to None.
            global_update (bool, optional): If True, update all of the particles
                as a single swarm. Defaults to False.
            verbose (bool, optional): Print out information. Defaults to True.
            n_swarms (int, optional): If global_update is False, it use the
                Swarm.n_swarms parameter. This value can be overwritten if
                desired by supplying it as an argument. This could be useful for
                strategies that enable small subswarms to communicate, e.g.
                initially have 2^n swarms, then after some iterations, change to
                2^(n-1) swarms for 1 or more iterations. This would propagate
                information between swarms without doing a full global update.
                Defaults to None.

        Returns:
            tuple: Tuple of numpy arrays containing the external and internal
                degrees of freedom
        """
        if result is not None:
            external = result["external"]
            internal = result["internal"]
            chi_2 = result["chi_2"]
            run = result["GALLOP Iter"]
        else:
            if external is None and internal is None:
                print("No DoFs supplied!")
                exit()
        self.position = self.get_position_from_dof(external, internal)
        if self.n_particles is None:
            self.n_particles = external.shape[0]

        if n_swarms is not None:
            self.n_swarms = n_swarms

        if self.particle_best_position is None:
            self.particle_best_position = np.copy(self.position)
            self.best_chi_2 = np.copy(chi_2)
        if self.velocity is None:
            self.velocity = np.zeros_like(self.position)
        self.update_best(chi_2)

        if not global_update:
            if self.global_update_freq is not None and self.global_update:
                if (run+1) % self.global_update_freq == 0 and run != 0:
                    global_update = True
        self.get_new_velocities(global_update=global_update, verbose=verbose)
        self.position = self.position + self.velocity

        if verbose:
            print(self.velocity.min(), self.velocity.max(),
            self.velocity.mean(),
            self.velocity.std(), np.abs(self.velocity).mean(),
            np.abs(self.velocity).std())

        external, internal = self.get_new_external_internal(self.position)

        return external, internal

    def get_CIF_of_best(self, n_reflections=None, one_for_each_subswarm=True,
                                filename_root=None, run=None, start_time=None):
        """
        Get a CIF of the best results found by the particle swarm

        Args:
            n_reflections (int, optional): The number of reflections used in the
                SDPD attempts. May be useful if comparing resolutions, but not
                normally needed. If None, then n_reflections = all reflections.
                Defaults to None.
            one_for_each_subswarm (bool, optional): A separate CIF for every
                independent subswarm rather than just the best globally.
                Defaults to True.
            filename_root (str, optional): Specify the root filename to use.
                If None, then use the structure name as the root.
                Defaults to None.
            run (int, optional): The GALLOP iteration. Defaults to None.
            start_time (float, optional): A float produced by time.time() that
                indicates when the run started. Defaults to None.
        """
        if not one_for_each_subswarm:
            external, internal = self.get_new_external_internal(
                                                    self.particle_best_position)
            chi_2 = self.best_chi_2
            external = external[chi_2 == chi_2.min()]
            internal = internal[chi_2 == chi_2.min()]
            if external.shape[0] > 1:
                external = external[0]
                internal = internal[0]
            chi_2 = chi_2.min()
            external = external.reshape(1,-1)
            internal = internal.reshape(1,-1)
        else:
            positions, chi2s = [], []
            for i in range(self.n_swarms):
                subswarm = self.n_particles // self.n_swarms
                begin = i*subswarm
                end = (i+1)*subswarm
                swarm_best_pos = self.particle_best_position[begin:end]
                swarm_chi2 = self.best_chi_2[begin:end]
                best_pos = swarm_best_pos[swarm_chi2 == swarm_chi2.min()]
                # If more than one particle has the same chi2, only save one
                # of them.
                if best_pos.shape[0] > 1:
                    best_pos = best_pos[0].reshape(1,-1)
                best_chi_2 = swarm_chi2.min()
                positions.append(best_pos)
                chi2s.append(best_chi_2)
            positions = np.vstack(positions)
            chi2s = np.array(chi2s)
            external, internal = self.get_new_external_internal(positions)
        if filename_root is None:
            filename_root = self.Structure.name
        for i in range(external.shape[0]):
            result = {}
            result["external"] = external[i]
            result["internal"] = internal[i]
            result["chi_2"] = chi2s[i]
            if run is None:
                result["GALLOP Iter"] = len(self.swarm_progress)
            else:
                result["GALLOP Iter"] = run
            if start_time is None:
                start_time = time.time()
            files.save_CIF_of_best_result(self.Structure, result, start_time,
                                    n_reflections, filename_root=filename_root
                                    +"_swarm_"+str(i))

    def reset_position_and_velocity(self):
        """
        Reset the Particle swarm
        """
        self.particle_best_position = None
        self.n_particles = None
        self.velocity = None
