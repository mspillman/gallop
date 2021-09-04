# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for local optimisation
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_optimizer as t_optim

from gallop import chi2
from gallop import tensor_prep
from gallop import files
from gallop import intensities
from gallop import zm_to_cart
from gallop.optim import restraints




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
            optimizer, loss, eps, save_CIF, streamlit, torsion_shadowing,
            Z_prime, use_restraints, include_PO, PO_axis
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
    settings["print_every"] = 10
    settings["check_min"] = 100
    settings["dtype"] = torch.float32
    settings["device"] = None
    settings["optimizer"] = "adam"
    settings["loss"] = "xlogx"
    settings["eps"] = 1e-8
    settings["save_CIF"] = True
    settings["streamlit"] = False
    settings["torsion_shadowing"] = False
    settings["Z_prime"] = 1
    settings["use_restraints"] = False
    settings["include_PO"] = False
    settings["PO_axis"] = [0, 0, 1]
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
    for i, param_group in enumerate(optimizer.param_groups):
        if i == 0:
            param_group['lr'] = lr
        else:
            # Slower updates for second parameter group, i.e. PO factor
            param_group['lr'] = lr/10
        if 'betas' in param_group.keys():
            if len(param_group['betas']) <= 2:
                param_group['betas'] = [b1, b2]
            else:
                param_group['betas'][0] = b1
                param_group['betas'][0] = b2
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
            Defaults to -4 (which is equivalent to 10^-4).
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
        if minimiser_settings["learning_rate_schedule"] == "constant":
            multiplication_factor *= 0.5
        minimum_point = trial_values[losses == losses.min()][0]
        return trial_values, losses, multiplication_factor, multiplication_factor * minimum_point
    else:
        print("ERROR! Minimiser Settings are needed!")
        return None

def minimise(Structure, external=None, internal=None, n_samples=10000,
    n_iterations=500, n_cooldown=100, device=None, dtype=torch.float32,
    n_reflections=None, learning_rate_schedule="1cycle",
    b_bounds_1_cycle=None, check_min=1, optimizer="Adam", verbose=False,
    print_every=10, learning_rate=3e-2, betas=None, eps=1e-8, loss="sum",
    start_time=None, run=1, torsion_shadowing=False, Z_prime=1,
    save_trajectories=False, save_grad=False, save_loss=False,
    include_dw_factors=True, chi2_solved=None, use_progress_bar=True,
    save_CIF=True, streamlit=False, use_restraints=False, include_PO=False,
    PO_axis=(0, 0, 1), notebook=False):
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
            to the intensities during optimization. Defaults to False.
        PO_axis (tuple, optional): The axis along which to apply the
            March-Dollase PO correction. Defaults to (0, 0, 1).
        notebook (bool, optional): If the code is running in a Jupyter notebook,
            use the tqdm notebook progress bars instead. Defaults to False.

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
        restraint_tensors = tensor_prep.get_restraint_tensors(Structure,
                                                                dtype, device)


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
        cosP, sinP, factor = tensor_prep.get_PO_tensors(Structure, PO_axis,
                            n_reflections, tensors["zm"]["external"].shape[0],
                            device, dtype)
        optimizer.add_param_group({"params" : [factor]})

    if start_time is None:
        t1 = time.time()
    else:
        t1 = start_time

    # Add the progress bar, if using
    if use_progress_bar and not streamlit:
        if notebook:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        iters = trange(n_iterations)
    else:
        iters = range(n_iterations)
        if streamlit:
            col1, col2, col3 = st.columns([10,1,1])
            with col1:
                prog_bar = st.progress(0.0)
            with col2:
                st.write(r"$\chi^{2}_{min}$")
            with col3:
                chi2_result = st.empty()

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

        if use_restraints:
            with torch.no_grad():
                min_chi_2 = chi_2#.min()
            restraint_penalty = restraints.get_restraint_penalties(
                                asymmetric_frac_coords, min_chi_2, 
                                **restraint_tensors)
        else:
            restraint_penalty = 0
        # PyTorch expects a single value for backwards pass.
        # Need a function to convert all of the chi_2 values into a scalar
        if isinstance(loss, str):
            if loss.lower() == "sse":
                L = ((chi_2 + restraint_penalty)**2).sum()
            elif loss.lower() == "sum":
                L = chi_2.sum() + restraint_penalty
            elif loss.lower() == "xlogx":
                L = torch.sum(torch.log(chi_2)*(chi_2 + restraint_penalty))
        else:
            if loss is None:
                # Default to the sum operation if loss is None
                L = chi_2.sum() + restraint_penalty
            else:
                try:
                    L = loss(chi_2, restraint_penalty)
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
            #prog_bar.progress(i/n_iterations)
            #with col1:
            prog_bar.progress(i/n_iterations)
            #with col2:
            if i % print_every == 0 or i == 1:
                with chi2_result:
                    st.write(str(np.around(chi_2.min().item(), 3)))
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

    # Now calculate the intensities with H-atoms included and use them to get a
    # profile chi2 estimate. If Structure.ignore_H_atoms is True, also calculate
    # the intensity chi2 with H-atoms included and add to the results dict.
    # This will also use the full data range even if n_reflections is set to a
    # subset of the available reflections.
    ignore_H_setting = Structure.ignore_H_atoms
    Structure.ignore_H_atoms = False
    best = result["chi_2"] == result["chi_2"].min()
    # Create tensors on CPU as it will be faster for a single data point.
    best_tensors = tensor_prep.get_all_required_tensors(Structure,
        external=result["external"][best][0].reshape(1,-1),
        internal=result["internal"][best][0].reshape(1,-1),
        requires_grad=False, device=torch.device("cpu"), verbose=False)
    # Restore the Structure.ignore_H_atoms setting
    Structure.ignore_H_atoms = ignore_H_setting

    with torch.no_grad():
        best_asym = zm_to_cart.get_asymmetric_coords(**best_tensors["zm"])

        best_intensities = intensities.calculate_intensities(best_asym,
                                                **best_tensors["int_tensors"])
        if include_PO:
            best_intensities = intensities.apply_MD_PO_correction(
                best_intensities, cosP.cpu(), sinP.cpu(),
                factor[best][0].cpu().reshape(1,1))

    if ignore_H_setting:
        chi_2_H = chi2.calc_chisqd(best_intensities,
                                            **best_tensors["chisqd_tensors"])

        result["best_chi_2_with_H"] = chi_2_H.detach().cpu().numpy()[0]
    else:
        result["best_chi_2_with_H"] = result["chi_2"].min()

    # Now get the profile chi2 if using DASH data
    if Structure.source.lower() == "dash":
        calc_profile = (best_intensities.cpu().numpy().reshape(
                        max(best_intensities.shape),1)
                    * Structure.baseline_peaks[:max(best.shape)]).sum(axis=0)
        sum1 = calc_profile[Structure.n_contributing_peaks != 0].sum()
        sum2 = Structure.profile[:,1][Structure.n_contributing_peaks != 0].sum()
        rescale = sum2/sum1
        subset = Structure.n_contributing_peaks != 0
        profchi2 = (((Structure.profile[:,2]**(-2))
            *(rescale*calc_profile - Structure.profile[:,1])**2)[subset].sum()
            / (subset.sum() - 2))
        result["prof_chi_2"] = profchi2
        result["calc_profile"] = rescale*calc_profile


    if save_trajectories:
        result["trajectories"] = trajectories

    if save_loss:
        result["losses"] = np.array(losses)

    if save_grad:
        result["gradients"] = gradients

    if include_PO:
        result["MD_factor"] = factor.detach().cpu().numpy()**2
        result["PO_axis"] = PO_axis
        del factor

    if save_CIF:
        files.save_CIF_of_best_result(Structure, result, start_time,
                                        n_reflections)
    del tensors
    return result