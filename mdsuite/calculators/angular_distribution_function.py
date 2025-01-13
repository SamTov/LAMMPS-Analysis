"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
Module to perform the calculation of the angular distribution function (ADF). The ADF
describes the average distribution of angles between three particles of species a, b,
and c. Note that a, b, and c may all be the same species, e.g. Na-Na-Na.
"""

import itertools
import logging
from abc import ABC
from dataclasses import dataclass
from typing import Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.calculators.calculator import call
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.utils.linalg import get_angles
from mdsuite.utils.meta_functions import join_path
from mdsuite.utils.neighbour_list import (
    get_neighbour_list,
    get_triplets,
    get_triu_indicies,
)

log = logging.getLogger(__name__)


@dataclass
class Args:
    """Data class for the saved properties."""

    number_of_bins: int
    number_of_configurations: int
    correlation_time: int
    atom_selection: np.s_
    data_range: int
    cutoff: float
    start: int
    norm_power: Union[int, float]
    stop: int
    species: list
    molecules: bool


class AngularDistributionFunction(TrajectoryCalculator, ABC):
    """
    Compute the Angular Distribution Function for all species combinations.

    Attributes
    ----------
    batch_size : int
        Number of batches, to split the configurations into.
    n_minibatches: int
        Number of minibatches for computing the angles to split each batch into
    n_confs: int
        Number of configurations to analyse
    r_cut: float
        cutoff radius for the ADF
    start: int
        Index of the first configuration
    stop: int
        Index of the last configuration
    bins: int
        bins for the ADF
    use_tf_function: bool, default False
        activate the tf.function decorator for the minibatches. Can speed up the
        calculation significantly, but may lead to excessive use of memory! During the
        first batch, this function will be traced. Tracing is slow, so this might only
        be useful for a larger number of batches.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.AngularDistributionFunction(n_confs = 100,
                                                           r_cut = 3.2,
                                                           batch_size = 10,
                                                           n_minibatches = 50,
                                                           start = 0,
                                                           stop = 200,
                                                           bins = 100,
                                                           use_tf_function = False)

    """

    def __init__(self, **kwargs):
        """
        Compute the Angular Distribution Function for all species combinations.

        Parameters
        ----------
        experiment : object
                Experiment object from which to take attributes.

        """
        super().__init__(**kwargs)
        self.scale_function = {"quadratic": {"outer_scale_factor": 10}}
        self.loaded_property = mdsuite_properties.positions

        self.use_tf_function = None
        self.molecules = None
        self.bin_range = None
        self.number_of_atoms = None
        self.norm_power = None
        self.sample_configurations = None
        self.result_keys = ["max_peak"]
        self.result_series_keys = ["angle", "adf"]
        self._dtype = tf.float32

        self.adf_minibatch = None  # memory management for triples generation per batch.

        self.analysis_name = "Angular_Distribution_Function"
        self.x_label = r"$$\text{Angle} / \theta$$"
        self.y_label = r"$$\text{ADF} / a.u.$$"

    @call
    def __call__(
        self,
        batch_size: int = 1,
        minibatch: int = -1,
        number_of_configurations: int = 5,
        cutoff: float = 6.0,
        start: int = 1,
        stop: int = None,
        number_of_bins: int = 500,
        species: list = None,
        use_tf_function: bool = False,
        molecules: bool = False,
        atom_selection=np.s_[:],
        plot: bool = True,
        norm_power: int = 4,
        **kwargs,
    ):
        """
        Parameters
        ----------
        batch_size : int
            Number of batches, to split the configurations into.
        minibatch: int
            Number of minibatches for computing the angles to split each batch into
        number_of_configurations: int
            Number of configurations to analyse
        cutoff: float
            cutoff radius for the ADF
        start: int
            Index of the first configuration
        stop: int
            Index of the last configuration
        number_of_bins: int
            bins for the ADF
        use_tf_function: bool, default False
            activate the tf.function decorator for the minibatches. Can speed
            up the calculation significantly, but may lead to excessive use of
            memory! During the first batch, this function will be traced.
            Tracing is slow, so this might only be useful for a larger number
            of batches.
        species : list
            A list of species to use.
        atom_selection : Union[np.s_, dict]
                Atoms to be used in the analysis.
        norm_power: int
            The power of the normalization factor applied to the ADF histogram.
            If set to zero no distance normalization will be applied.
        molecules : bool
                if true, perform the analysis on molecules.
        plot : bool
                If true, plot the result of the analysis.

        """
        # set args that will affect the computation result
        self.args = Args(
            number_of_bins=number_of_bins,
            cutoff=cutoff,
            start=start,
            stop=stop,
            atom_selection=atom_selection,
            data_range=1,
            correlation_time=1,
            molecules=molecules,
            species=species,
            number_of_configurations=number_of_configurations,
            norm_power=norm_power,
        )

        # Parse the user arguments.
        self.use_tf_function = use_tf_function
        self.cutoff = cutoff
        self.plot = plot
        self._batch_size = batch_size  # memory management for all batches
        self.adf_minibatch = minibatch
        self.bin_range = [0.0, 3.15]  # from 0 to a chemists pi
        self.norm_power = norm_power
        self.override_n_batches = kwargs.get("batches")

    def check_input(self):
        """
        Check the inputs and set defaults if necessary.

        Returns
        -------
        Updates the class attributes.

        """
        self._run_dependency_check()
        if self.args.stop is None:
            self.args.stop = self.experiment.number_of_configurations - 1

        # Get the correct species out.
        if self.args.species is None:
            if self.args.molecules:
                self.args.species = list(self.experiment.molecules)
                self._compute_number_of_atoms(self.experiment.molecules)
            else:
                self.args.species = list(self.experiment.species)
                self._compute_number_of_atoms(self.experiment.species)

        else:
            self._compute_number_of_atoms(self.experiment.species)

    def _compute_number_of_atoms(self, reference: dict):
        """
        Compute the number of atoms total in the selected set of species.

        Parameters
        ----------
        reference : dict
                Reference dictionary in which to look for indices lists

        Returns
        -------
        Updates the number of atoms attribute.

        """
        # TODO return to dotdict form when molecules is a dotdict
        number_of_atoms = 0
        for item in self.args.species:
            if isinstance(self.args.atom_selection, dict):
                number_of_atoms = 0
                for item in self.args.atom_selection:
                    number_of_atoms += len(self.args.atom_selection[item])
            else:
                number_of_atoms += reference[item].n_particles

        self.number_of_atoms = number_of_atoms

        if self.adf_minibatch == -1:
            self.adf_minibatch = number_of_atoms

    def _prepare_data_structure(self):
        """
        Prepare variables and dicts for the analysis.

        Returns
        -------
        -------.

        """
        sample_configs = np.linspace(
            self.args.start,
            self.args.stop,
            self.args.number_of_configurations,
            dtype=int,
        )

        species_indices = []
        start_index = 0
        stop_index = 0
        for species in self.args.species:
            try:
                if isinstance(self.args.atom_selection, dict):
                    stop_index += len(self.args.atom_selection[species])
                else:
                    stop_index += self.experiment.species[species].n_particles
            except KeyError:
                stop_index += self.experiment.molecules[species].n_particles
            species_indices.append((species, start_index, stop_index))
            start_index = stop_index

        return sample_configs, species_indices

    def _prepare_triples_generator(self):
        """
        Prepare the triples generators including tf.function
        Returns
        -------.

        """
        if self.use_tf_function:

            @tf.function
            def _get_triplets(x):
                return get_triplets(
                    x,
                    r_cut=self.cutoff,
                    n_atoms=self.number_of_atoms,
                    n_batches=self.adf_minibatch,
                    disable_tqdm=True,
                )

        else:

            def _get_triplets(x):
                return get_triplets(
                    x,
                    r_cut=self.cutoff,
                    n_atoms=self.number_of_atoms,
                    n_batches=self.adf_minibatch,
                    disable_tqdm=True,
                )

        return _get_triplets

    def _compute_rijk_matrices(self, tmp: tf.Tensor, timesteps: int):
        """
        Compute the rij matrix.

        Returns
        -------

        """
        _get_triplets = self._prepare_triples_generator()

        r_ij_flat = next(
            get_neighbour_list(tmp, cell=self.experiment.box_array, batch_size=1)
        )

        r_ij_indices = get_triu_indicies(self.number_of_atoms)
        # Shape is now (n_atoms, n_atoms, 3, n_timesteps)
        r_ij_mat = tf.scatter_nd(
            indices=tf.transpose(r_ij_indices),
            updates=tf.transpose(r_ij_flat, (1, 2, 0)),
            shape=(self.number_of_atoms, self.number_of_atoms, 3, timesteps),
        )

        r_ij_mat -= tf.transpose(r_ij_mat, (1, 0, 2, 3))
        r_ij_mat = tf.transpose(r_ij_mat, (3, 0, 1, 2))

        r_ijk_indices = _get_triplets(r_ij_mat)

        return r_ij_mat, r_ijk_indices

    @staticmethod
    def _compute_angles(species, r_ijk_indices):
        """
        Compute the angles between indices in triangle.

        Parameters
        ----------
        species
        r_ijk_indices

        Returns
        -------
        condition
        name

        """
        (i_name, i_min, i_max), (j_name, j_min, j_max), (k_name, k_min, k_max) = species
        name = f"{i_name}-{j_name}-{k_name}"

        i_condition = tf.logical_and(
            r_ijk_indices[:, 1] >= i_min, r_ijk_indices[:, 1] < i_max
        )
        j_condition = tf.logical_and(
            r_ijk_indices[:, 2] >= j_min, r_ijk_indices[:, 2] < j_max
        )
        k_condition = tf.logical_and(
            r_ijk_indices[:, 3] >= k_min, r_ijk_indices[:, 3] < k_max
        )

        condition = tf.math.logical_and(
            x=tf.math.logical_and(x=i_condition, y=j_condition), y=k_condition
        )

        return condition, name

    def _build_histograms(self, positions, species_indices, angles):
        """
        Build the adf histograms.

        Returns
        -------
        angles : dict
                A dictionary of the triples references and their histogram values.

        """
        tmp = tf.transpose(tf.concat(positions, axis=0), (1, 0, 2))

        timesteps, atoms, _ = tf.shape(tmp)

        r_ij_mat, r_ijk_indices = self._compute_rijk_matrices(tmp, timesteps)

        for species in itertools.combinations_with_replacement(species_indices, 3):
            # Select the part of the r_ijk indices, where the selected species
            # triple occurs.
            condition, name = self._compute_angles(species, r_ijk_indices)
            tmp = tf.gather_nd(r_ijk_indices, tf.where(condition))

            # Get the indices required.
            angle_vals, pre_factor = get_angles(r_ij_mat, tmp)
            pre_factor = 1 / pre_factor**self.norm_power
            histogram, _ = np.histogram(
                angle_vals,
                bins=self.args.number_of_bins,
                range=self.bin_range,
                weights=pre_factor,
                density=True,
            )
            histogram = tf.cast(histogram, dtype=tf.float32)
            if angles.get(name) is not None:
                angles.update({name: angles.get(name) + histogram})
            else:
                angles.update({name: histogram})

        return angles

    def _compute_adfs(self, angles, species_indices):
        """
        Compute the ADF and store it.

        Parameters
        ----------
        angles : dict
                Dict of angle combinations and their histograms.
        species_indices : list
                A list of indices associated with species under consideration.

        Returns
        -------
        Updates the class, the SQL database, and plots values if required.

        """
        for species in itertools.combinations_with_replacement(species_indices, 3):
            name = f"{species[0][0]}-{species[1][0]}-{species[2][0]}"
            hist = angles.get(name)

            bin_range_to_angles = np.linspace(
                self.bin_range[0] * (180 / 3.14159),
                self.bin_range[1] * (180 / 3.14159),
                self.args.number_of_bins,
            )

            self.selected_species = [_species[0] for _species in species]

            self.data_range = self.args.number_of_configurations
            log.debug(f"species are {species}")
            max_angle = bin_range_to_angles[tf.math.argmax(hist.numpy())]
            data = {
                self.result_keys[0]: max_angle,
                self.result_series_keys[0]: bin_range_to_angles.tolist(),
                self.result_series_keys[1]: hist.numpy().tolist(),
            }

            self.queue_data(data=data, subjects=self.selected_species)

    def plot_data(self, data):
        """Plot data."""
        for selected_species, val in data.items():
            bin_range_to_angles = np.linspace(
                self.bin_range[0] * (180 / 3.14159),
                self.bin_range[1] * (180 / 3.14159),
                len(val[self.result_series_keys[0]]),
            )
            title_value = bin_range_to_angles[
                tf.math.argmax(val[self.result_series_keys[1]])
            ]

            self.run_visualization(
                x_data=np.array(val[self.result_series_keys[0]]),
                y_data=np.array(val[self.result_series_keys[1]]),
                title=f"{selected_species} - Max: {title_value:.3f} degrees ",
            )

    def _format_data(self, batch: tf.Tensor, keys: list) -> tf.Tensor:
        """
        Format the loaded data for use in the rdf calculator.

        The RDF requires a reshaped dataset. The generator will load a default
        dict oriented type. This method restructures the data to be used in the
        calculator.

        Parameters
        ----------
        batch : tf.Tensor
                A batch of data to transform.
        keys : list
                Dict keys to extract from the data.

        Returns
        -------

        """
        formatted_data = []
        for item in keys:
            formatted_data.append(batch[item])

        if len(self.args.species) == 1:
            return tf.cast(formatted_data[0], dtype=self.dtype)
        else:
            return tf.cast(tf.concat(formatted_data, axis=0), dtype=self.dtype)

    def _correct_batch_properties(self):
        """
        We must fix the batch size parameters set by the parent class.

        Returns
        -------
        Updates the parent class.

        """
        if self.batch_size > self.args.number_of_configurations:
            self.batch_size = self.args.number_of_configurations
            self.n_batches = 1
        else:
            self.n_batches = int(self.args.number_of_configurations / self.batch_size)

        if self.override_n_batches is not None:
            self.n_batches = self.override_n_batches

        if self.minibatch:
            self.batch_size = 1
            self.n_batches = self.args.number_of_configurations
            self.memory_manager.atom_batch_size = None
            self.memory_manager.n_atom_batches = None
            self.memory_manager.atom_remainder = None
            self.minibatch = False

        self.remainder = 0
        self.minibatch = False

    def prepare_computation(self):
        """
        Run steps necessary to prepare the computation for running.

        Returns
        -------

        """
        path_list = [
            join_path(species_name, self.loaded_property.name)
            for species_name in self.args.species
        ]
        self._prepare_managers(path_list)

        # batch loop correction
        self._correct_batch_properties()

        # Get the correct dict keys.
        dict_keys = []
        for species_name in self.args.species:
            dict_keys.append(
                str.encode(join_path(species_name, self.loaded_property.name))
            )

        # Split the configurations into batches.
        split_arr = np.array_split(self.sample_configurations, self.n_batches)

        return dict_keys, split_arr

    def run_calculator(self):
        """
        Run the analysis.

        Returns
        -------

        """
        self.check_input()
        self.sample_configurations, species_indices = self._prepare_data_structure()

        dict_keys, split_arr = self.prepare_computation()

        # Get the batch dataset
        batch_ds = self.get_batch_dataset(
            subject_list=self.args.species, loop_array=split_arr, correct=True
        )

        angles = {}

        # Loop over the batches.
        for idx, batch in tqdm(enumerate(batch_ds), ncols=70, total=self.n_batches):
            positions_tensor = self._format_data(batch=batch, keys=dict_keys)

            angles = self._build_histograms(
                positions=positions_tensor,
                species_indices=species_indices,
                angles=angles,
            )

        self._compute_adfs(angles, species_indices)
