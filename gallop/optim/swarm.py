# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides a class for particle swarm.
"""

import time
import tqdm
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import qmc

from gallop import files



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

    def get_initial_positions(self, method="latin", MDB=None):
        """
        Generate the initial starting points for a GALLOP attempt. The
        recommended method uses latin hypercube sampling which provides a
        more even coverage of the search space than random uniform sampling,
        which can produce clusters or leave regions unexplored.

        Args:
            method (str, optional): The sampling method to use. Can be one of
                "latin" or "uniform". Defaults to "latin".
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
        subswarm = int(self.n_particles // self.n_swarms)
        init_external = []
        init_internal = []

        total_pos = self.Structure.total_position_degrees_of_freedom
        total_rot = self.Structure.total_rotation_degrees_of_freedom
        tot_external = total_pos+total_rot
        total_tors = self.Structure.total_internal_degrees_of_freedom
        # Separate hypercube for each subswarm
        for _ in tqdm.tqdm(range(int(self.n_swarms))):
            if method == "latin":
                lhc = qmc.LatinHypercube(int(total_pos+total_rot+total_tors))
                all_dof = lhc.random(subswarm)
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
        chi_2=None, run=None, global_update=False, verbose=False, n_swarms=None):
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
            verbose (bool, optional): Print out information. Defaults to False.
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
            chi2s = [chi_2.min()]
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
