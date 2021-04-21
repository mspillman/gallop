import torch
import gallop.chi2 as chi2
import gallop.tensor_prep as tensor_prep
import gallop.files as files
import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt
import os
import random
#from SALib.sample import latin
import pyDOE
import torch_optimizer as t_optim

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


def seed_everything(seed=1234, change_backend=True):
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
            ignore_reflections_for_chi2_calc, optimizer, loss, eps, save_CIF
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
    return settings


def adjust_lr_1_over_sqrt(optimizer, iteration, learning_rate):
    """
    Decay the learning_rate used in the local optimizer by 1/sqrt(iteration+1)
    Called by the minimise function

    Args:
        optimizer (pytorch optimizer): A pytorch compatible optimizer
        iteration (int): The current iteration
        learning_rate (float): The base-learning rate to use

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
    low = lowest learning rate
    high = highest learning rate
    final = iteration when final lr decrease occurs
    """

    increment = (high-low)/(final//2)
    b1_increment = (upperb1 - lowb1) / (final//2)
    b2_increment = (upperb2 - lowb2) / (final//2)
    final_increment = low/(num_iterations-final)
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

def find_learning_rate(Structure, external=None, internal=None, n_samples=10000,
    n_iterations=500, n_cooldown=100, device=None, dtype=torch.float32,
    n_reflections=None, min_lr=-4, max_lr=0, n_trials=200,
    multiplication_factor=1., optimizer="Adam", verbose=True, amsgrad=False,
    learning_rate=1e-1, betas=[0.9,0.9], loss="sum", plot=True, logplot=True,
    use_progress_bar=True, eps=1e-8, ignore_reflections_for_chi2_calc=False,
    include_dw_factors=True,
    b_bounds_1_cycle = {"upperb1":0.95, "lowb1":0.85, "upperb2":0.9,
    "lowb2":0.9}, chi2_solved=None, learning_rate_schedule="array",
    print_every=100, check_min=1, save_CIF=False, figsize=(6,6),
    streamlit=False):
    """
    See here: https://sgugger.github.io/the-1cycle-policy.html
    In contrast to the advice in the article (which is for training neural
    networks), in this work, it seems to work better to take the absolute
    minimum point that is obtained.

    Most arguments are the same as that used for minimise (see docstring)
    The ones unique to this function are detailed below
    Args:
        min_lr (int, optional): Minimum (log10) learning rate to try.
            Defaults to -5 (which is equivalent to 10^-4).
        max_lr (int, optional): Maximum (log10) learning rate to try.
            Defaults to 0 (which is equivalent to 10^-0.5).
        n_trials (int, optional): Number of trials between min_lr and max_lr.
            Defaults to 200.
        multiplication_factor (float, optional): multiply the lowest points
            in the learning rate curve by a fixed amount, e.g. 0.5 or 2.0 to
            decrease and increase the learning rate respectively. Defaults to 1.
        plot (bool, optional): Plot the curve of learning rate vs loss.
            Defaults to True.
        logplot (bool, optional): If plotting, use a logarithmic x-axis.
            Defaults to True.

    Returns:
        tuple: Tuple containing the trial learning rate values, the losses
                obtained and the learning rate associated with the minimum
                loss value
    """

    # Set the learning rates to be tested
    trial_values = np.logspace(min_lr, max_lr, n_trials)

    # Get the losses at each learning rate
    result = minimise(Structure, run=-1, learning_rate_schedule="array",
        learning_rate=trial_values, external=external,
        internal=internal, n_samples=n_samples,
        n_iterations=n_trials, n_cooldown=n_cooldown, device=device,
        dtype=dtype, n_reflections=n_reflections,
        b_bounds_1_cycle=b_bounds_1_cycle, optimizer=optimizer,
        verbose=False, betas=betas, eps=eps, loss=loss,
        save_trajectories=False, save_loss=True,
        include_dw_factors=include_dw_factors,
        ignore_reflections_for_chi2_calc=ignore_reflections_for_chi2_calc,
        use_progress_bar=use_progress_bar, save_CIF=False, streamlit=streamlit)

    losses = result["losses"]


    if plot:
        plt.figure(figsize=figsize)
        plt.plot(trial_values, losses)
        if logplot:
            plt.xscale('log')
        plt.show()

    minimum_point = trial_values[losses == losses.min()][0]
    return trial_values, losses, multiplication_factor * minimum_point


def minimise(Structure, external=None, internal=None, n_samples=10000,
    n_iterations=500, n_cooldown=100, device=None, dtype=torch.float32,
    n_reflections=None, learning_rate_schedule="1cycle",
    b_bounds_1_cycle = {"upperb1":0.95, "lowb1":0.85, "upperb2":0.9,
    "lowb2":0.9}, check_min=1, optimizer="Adam", verbose=False, print_every=100,
    learning_rate=3e-2, betas=[0.9,0.9], eps=1e-8, loss="sum", start_time=None,
    run=1, save_trajectories=False, save_grad=False, save_loss=False,
    include_dw_factors=True, chi2_solved=None,
    ignore_reflections_for_chi2_calc=False, use_progress_bar=True,
    save_CIF=True, streamlit=False):
    """
    Main minimiser function used by GALLOP. Take a set of input external and
    internal degrees of freedom together with the observed intensities and
    inverse covariance matrix, and optimise the chi-squared factor of agreement
    between the calculated and observed intensities.

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
            The number of iterations at the end of a local optimsation run with
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
                            rates
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
            policy to set the upper and lower beta values.
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
            to [0.9, 0.9] which is equivalent to [beta1=0.9, beta2=0.9]
        eps (float, optional): Epsilon value to use in Adam or Adam derived
            optimizers. Defaults to 1e-8.
        loss (str or function, optional):   Pytorch requires a scalar to
            determine the gradients so the series of chi2 values obtained from
            the n_samples must be combined through some function before the
            gradient is determined. If string, must be one of:
            sum, sse, xlogx
                sum     -   add all the chi2 values together
                sse     -   sum(chi2^2) (sum of squared errors)
                xlogx   -   sum(chi2 * log(chi2))
            If a function, must take in a pytorch tensor and return a scalar.
            Defaults to "sum".
        start_time (time.time(), optional): The start time of a run or set of
            runs. Useful in full GALLOP runs, but if set to None then the start
            time will automatically be determined.
        run (int, optional): If a set of runs are being performed, then the run
            number can be passed for printing. Defaults to 1.
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
            this set to True enables pretty print outputs etc. Defaults to False

    Returns:
        dictionary: A dictionary containing the optimised external and internal
            degrees of freedom and their associated chi_2 values.
            If save_trajectories is True then this also contains the
            trajectories of the particles, the chi_2 values and loss values
            at each iteration.
    """

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

    # Initialize the optimizer
    if isinstance(optimizer, str):
        if learning_rate_schedule.lower() == "array":
            init_lr = learning_rate[0]
        else:
            init_lr = learning_rate
        if optimizer.lower() == "adam":
            optimizer = torch.optim.Adam([tensors["external"],
                                        tensors["internal"]],
                                        lr=init_lr, betas=betas, eps=eps)
        elif optimizer.lower() == "yogi":
            optimizer = t_optim.Yogi([tensors["external"],
                                        tensors["internal"]],
                                        lr=init_lr, betas=betas, eps=eps)
        elif optimizer.lower() == "diffgrad":
            optimizer = t_optim.DiffGrad([tensors["external"],
                                        tensors["internal"]],
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
            optimizer.param_groups[0]["params"] = [tensors["external"],
                                                    tensors["internal"]]
            for param_group in optimizer.param_groups:
                if learning_rate_schedule.lower() == "array":
                    param_group['lr'] = learning_rate[0]
                else:
                    param_group['lr'] = learning_rate

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
            except:
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
                print("An error has occured with lr scheduling")

        # Forward pass - this gets a tensor of shape (n_samples, 1) with a
        # chi_2 value for each set of external/internal DoFs.
        chi_2 = chi2.get_chi_2(**tensors)
        if ignore_reflections_for_chi2_calc:
            # Counteract the division normally used in chi2 calculation
            chi_2 *= (tensors["hkl"].shape[1] - 2)

        # PyTorch expects a single value for backwards pass.
        # Need a function to convert all of the chi_2 values into a scalar
        if isinstance(loss, str):
            if loss.lower() == "sse":
                L = (chi_2**2).sum()
            elif loss.lower() == "sum":
                L = chi_2.sum()
            elif loss.lower() == "xlogx":
                L = torch.sum(chi_2*torch.log(chi_2))
        else:
            if loss is None:
                # Default to the sum operation if loss is None
                L = chi_2.sum()
            else:
                try:
                    L = loss(chi_2)
                except:
                    print("Unknown / incompatible loss function",loss)
                    print("Allowable arguments = sse (sum of squared errors),",
                        "sum, xlogx or a suitable function that returns a",
                        "single scalar value")

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
                    printstring = ("GALLOP iter {:04d} | LO iter {:04d} | lr {:.3f} ||",
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
                    printstring = ("GALLOP iter {:04d} | LO iter {:04d} | lr {:.3f} ||",
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
            trajectories.append([tensors["external"].detach().cpu().numpy(),
                                tensors["internal"].detach().cpu().numpy(),
                                chi_2.detach().cpu().numpy(),
                                L.detach().cpu().numpy()])
        if save_grad:
            gradients.append([tensors["external"].grad.detach().cpu().numpy(),
                            tensors["internal"].grad.detach().cpu().numpy()])
        if i != n_iterations:
            optimizer.step()
        if streamlit:
            prog_bar.progress(i/n_iterations)
    result = {
            "external"     : tensors["external"].detach().cpu().numpy(),
            "internal"     : tensors["internal"].detach().cpu().numpy(),
            "chi_2"        : chi_2.detach().cpu().numpy(),
            "GALLOP Iter"  : run
            }
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
    verbose=False, figsize=(10,10), xlim={"left" : 0, "right" : None},
    ylim={"bottom" : 0, "top" : None}, cmap="tab20", call_show=True):
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
        position = None, best_subswarm_chi_2 = None, inertia="ranked", c1=1.5,
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
            best_subswarm_chi_2 (list, optional): The best chi_2 found in each
                subswarm. Defaults to None.
            inertia (float or str, optional): The inertia to use in the velocity
                update. If random, sample the inertia from a uniform
                distribution. If "ranked", then solutions ranked in order of
                increasing chi2. Lowest chi2 assigned lowest inertia, as defined by
                bounds in inertia_bounds. Defaults to "ranked".
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
                    run number % global_update_freq == 0
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
        self.best_subswarm_chi_2 = best_subswarm_chi_2
        self.inertia=inertia
        self.c1=c1
        self.c2=c2
        self.inertia_bounds=inertia_bounds
        self.use_matrix=use_matrix
        self.swarm_progress = []
        self.limit_velocity = limit_velocity
        self.n_particles = n_particles
        self.best_low_res_chi_2 = None
        self.best_high_res_chi_2 = None
        self.global_update = global_update
        self.global_update_freq = global_update_freq
        self.vmax = vmax

    def get_initial_positions(self, method="latin", latin_criterion=None):
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

        Returns:
            tuple: Tuple of numpy arrays containing the initial external and
            internal degrees of freedom
        """
        if not hasattr(self.Structure, "total_internal_degrees_of_freedom"):
            self.Structure.get_total_degrees_of_freedom()

        assert method in ["uniform", "latin"], "method must be latin or uniform"
        if self.n_particles % self.n_swarms != 0:
            print("n_particles should be divisible by n_swarms.")
            self.n_particles = n_swarms * (n_particles // n_swarms)
            print("Setting n_particles to", self.n_particles)
        subswarm = self.n_particles // self.n_swarms
        init_external = []
        init_internal = []

        total_pos = self.Structure.total_position_degrees_of_freedom
        total_rot = self.Structure.total_rotation_degrees_of_freedom
        tot_external = total_pos+total_rot
        total_tors = self.Structure.total_internal_degrees_of_freedom
        ext_names = ["ext"+str(x) for x in range(total_pos+total_rot)]
        int_names = ["int"+str(x) for x in range(total_tors)]
        for i in tqdm.tqdm(range(self.n_swarms)):
            if method == "latin":
                #SAlib old latin code
                #problem = {
                #"num_vars" : total_pos + total_rot + total_tors,
                #"names": ext_names + int_names,
                #"bounds" : (total_pos*[[0,1]]
                #            + total_rot*[[-1,1]]
                #            + total_tors*[[-np.pi, np.pi]])
                #}
                #init_dof = latin.sample(problem, subswarm)
                #init_external.append(init_dof[:,:total_pos+total_rot])
                #init_internal.append(init_dof[:,total_pos+total_rot:])

                #Try pyDOE as alternative library
                all_dof = np.array(pyDOE.lhs(total_pos + total_rot + total_tors,
                            samples=subswarm, criterion=latin_criterion))
                external = all_dof[:,:total_pos+total_rot]
                pos = external[:,:total_pos]
                rot = external[:,total_pos:]
                tor = all_dof[:,total_pos+total_rot:]
                rot -= 0.5
                rot *= 2 # Rotation to range [-1,1]
                tor -= 0.5
                tor *= 2 * np.pi # Torsions to range [-pi,pi]
                init_external.append(np.hstack([pos,rot]))
                init_internal.append(tor)

            else:
                rand_ext = np.random.uniform(-1,1,size=(subswarm,tot_external))
                rand_int = np.random.uniform(-1,1,size=(subswarm,total_tors))
                init_external.append(rand_ext)
                init_internal.append(rand_int)

        init_external = np.vstack(init_external)
        init_internal = np.vstack(init_internal)
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
        translation = translation % 1   # Convert into range(0,1)
        translation *= 2 * np.pi         # Convert into range(0, 2pi)
        translation = np.hstack([np.sin(translation), np.cos(translation)])

        rotation = np.copy(external[:,end_of_translations:])
        rotation_list = []
        for i in range(n_quaternions):
            # Ensure quaternions are unit quaternions
            quaternion = rotation[:,(i*4):(i+1)*4]
            quaternion /= np.sqrt((quaternion**2).sum(axis=1)).reshape(-1,1)
            # Now extract angle and unit-axis representation of quaternion, and
            # project angle onto unit circle
            #angle = 2*np.arccos(quaternion[:,0]).reshape(-1,1)
            #axis = quaternion[:,1:]
            #axis /= np.sqrt((axis**2).sum(axis=1)).reshape(-1,1)
            #rotation_list.append(np.hstack([np.sin(angle), np.cos(angle), axis]))
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

        Returns:
            tuple: Tuple of numpy arrays containing the external and internal
                degrees of freedom
        """
        total_external = self.Structure.total_external_degrees_of_freedom
        total_position = self.Structure.total_position_degrees_of_freedom
        total_rotation = self.Structure.total_rotation_degrees_of_freedom
        total_torsional = self.Structure.total_internal_degrees_of_freedom
        # Recall, we store an extra component so the swarm representation of a
        # quaternion has 5 components rather than 4
        n_quaternions = total_rotation // 4
        end_external = (2*total_position) + total_rotation #+ n_quaternions
        external = np.copy(position[:,:end_external])
        internal = np.copy(position[:,end_external:])
        # Reverse the normalisation of the particle position,
        # back to range 0 - 1
        pos_sines = external[:,:total_position]
        pos_cosines = external[:,total_position:2*total_position]
        # Can now use the inverse tangent to get positions in range -0.5, 0.5
        translations = np.arctan2(pos_sines, pos_cosines) / (2*np.pi)
        #translations = translations % 1 # Return translations to the range(0, 1)

        rotations = external[:,2*total_position:]
        # Ensure the quaternions are unit quaternions

        rotation_list = []
        for i in range(n_quaternions):
            #angle = np.arctan2(rotations[:,(i*5):(i+1)*5][:,0],
            #                    rotations[:,(i*5):(i+1)*5][:,1])
            #axis = rotations[:,(i*5):(i+1)*5][:,2:]
            #axis /= np.sqrt((axis**2).sum(axis=1)).reshape(-1,1)
            #cos_angle = np.cos(angle/2).reshape(-1,1)
            #sin_angle = np.sin(angle/2).reshape(-1,1)
            #quaternion = np.hstack([cos_angle, sin_angle*axis])
            quaternion = rotations[:,(i*4):(i+1)*4]
            quaternion /= np.sqrt((quaternion**2).sum(axis=1)).reshape(-1,1)
            rotation_list.append(quaternion)
        rotations = np.hstack(rotation_list)

        external = np.hstack([translations, rotations])
        # Revert torsion angles back to angles using the inverse tangent
        internal = np.arctan2(internal[:,:total_torsional],
                            internal[:,total_torsional:])

        return external, internal

    def PSO_velocity_update(self, previous_velocity, position,
        particle_best_pos, best_chi_2, inertia="random", c1=1.5, c2=1.5,
        inertia_bounds=[0.4,0.9], use_matrix=True):
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
                If "ranked", then set the inertia values linearaly between the
                bounds, with the lowest inertia for the best particle. If a
                float, then all particles are assigned the same inertia.
                Defaults to "random".
            c1 (int, optional): c1 (social) parameter in PSO equation.
                Defaults to 1.5.
            c2 (int, optional): c2 (cognitive) parameter in PSO equation.
                Defaults to 1.5.
            inertia_bounds (list, optional): The upper and lower bound of the
                values that inertia can take if inertia is set to "random" or
                "ranked". Defaults to [0.4,0.9].
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
                print("Setting intertia to 0.5")
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

    def get_new_velocities(self, global_update = True, verbose = True):
        """
        Update the particle velocities using the PSO equations.
        Can either update all particles as a single swarm, or treat them as a
        set of independent swarms (or subswarms).

        Args:
            global_update (bool, optional): If True, update all of the particles
                as a single swarm. If False, then update a total of n_swarms
                separately. Defaults to True.
            verbose (bool, optional): Print out information. Defaults to True.
        """
        self.best_subswarm_chi2 = []
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
                self.best_subswarm_chi2.append(swarm_chi2.min())
            self.swarm_progress.append(self.best_subswarm_chi2)

        if self.limit_velocity:
            unlimited = self.velocity
            #self.velocity[unlimited >= 0] = unlimited[unlimited >= 0] % 2
            #self.velocity[unlimited < 0] = unlimited[unlimited < 0] % -2
            self.velocity[unlimited > self.vmax] = self.vmax
            self.velocity[unlimited < -1*self.vmax] = -1*self.vmax

    def update_position(self, result=None, external=None, internal=None,
        chi_2=None, run=None, global_update=False, verbose=True, n_swarms=None):
        """
        Take a set of results from the minimisation algorithm and use
        them to generate a new set of starting points to be minimised using
        the particle swarm algorithm.

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
                Swarm object initialisation n_swarms parameter. This value can
                be overwritten if desired by supplying it as an argument.
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
        if not one_for_each_subswarm:
            external, internal = self.get_new_external_internal(
                                                    self.particle_best_position)
            chi_2 = self.chi_2
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
            result["external"] = external[i]#.reshape(1,-1)
            result["internal"] = internal[i]#.reshape(1,-1)
            result["chi_2"] = chi2s[i]
            result["GALLOP Iter"] = len(self.swarm_progress)
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

def resolution_switch(run, minimiser_settings,
    Swarm, external, internal, result=None, switch_freq=10, switch_factor=2,
    find_learning_rate_on_switch=True, low_res_start=False, keep_pos=False,
    keep_vel=False, verbose=True, find_lr_kw_args={}):

    n_reflections = minimiser_settings["n_reflections"]
    if n_reflections is None:
        n_reflections = len(Swarm.Structure.hkl)
    elif n_reflections > len(Swarm.Structure.hkl):
        n_reflections = len(Swarm.Structure.hkl)
    switch = False
    if (run+1) % switch_freq == 0 and (run+1) % (switch_freq*2) != 0 and run != 0:
        if low_res_start:
            n_reflections = int(n_reflections*switch_factor)
        else:
            n_reflections = int(n_reflections/switch_factor)
        switch = True
    elif (run+1) % (switch_freq*2) == 0 and run != 0:
        if low_res_start:
            n_reflections = int(n_reflections/switch_factor)
        else:
            n_reflections = int(n_reflections*switch_factor)
        switch = True

    if switch:
        minimiser_settings["n_reflections"] = n_reflections
        if verbose:
            print("Setting n_reflections to:", n_reflections)
            print("Resolution with {} reflections:".format(n_reflections),
                Swarm.Structure.get_resolution(
                                    Swarm.Structure.twotheta[n_reflections-1]))
        if find_learning_rate_on_switch:
            if result is not None:
                lr = find_learning_rate(Swarm.Structure,
                            external=result["external"],
                            internal=result["internal"],
                            **find_lr_kw_args, **minimiser_settings)
                learning_rate = lr[-1]
                if verbose:
                    print("Setting learning rate to:", learning_rate)
                minimiser_settings["learning_rate"] = learning_rate
            else:
                lr = find_learning_rate(Swarm.Structure, external=external,
                                        internal=internal,
                                        **find_lr_kw_args, **minimiser_settings)
                learning_rate = lr[-1]
                if verbose:
                    print("Setting learning rate to:", learning_rate)
                minimiser_settings["learning_rate"] = learning_rate
        if not keep_vel:
            Swarm.velocity = None
            if verbose:
                print("Reset particle velocities")
        if not keep_pos:
            Swarm.particle_best_position = None
            if verbose:
                print("Reset best particle positions")


