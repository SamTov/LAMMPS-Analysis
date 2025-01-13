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
MDSuite module for the computation of the radial distribution function (RDF). An RDF
describes the probability of finding a particle of species b at a distance r of
species a.
"""

from __future__ import annotations

import itertools
import logging
from abc import ABC
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Union

import numpy as np
import tensorflow as tf

# Import user packages
from tqdm import tqdm

from mdsuite.calculators.calculator import call
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.utils.linalg import (
    apply_minimum_image,
    apply_system_cutoff,
    get_partial_triu_indices,
)
from mdsuite.utils.meta_functions import join_path, split_array

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
    stop: int
    species: list
    molecules: bool


class RadialDistributionFunction(TrajectoryCalculator, ABC):
    """
    Class for the calculation of the radial distribution function.

    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    data_range :
            Number of configurations to use in each ensemble
    x_label : str
            X label of the tensor_values when plotted
    y_label : str
            Y label of the tensor_values when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database_path for the analysis
    minibatch: int, default None
            Size of an individual minibatch, if set. By default mini-batching is not
            applied

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    .. code-block:: python

        project = mdsuite.Project()
        project.run.RadialDistributionFunction(number_of_configurations=500)

    """

    def __init__(self, **kwargs):
        """
        Constructor for the RDF calculator.

        Attributes
        ----------
        kwargs: see RunComputation class for all the passed arguments

        """
        super().__init__(**kwargs)

        self.scale_function = {
            "quadratic": {"outer_scale_factor": 10, "inner_scale_factor": 5}
        }
        self.loaded_property = mdsuite_properties.positions
        self.x_label = r"$$r / nm$$"
        self.y_label = r"$$g(r)$$"
        self.analysis_name = "Radial_Distribution_Function"
        self.result_series_keys = ["x", "y"]

        self._dtype = tf.float32

        self.rdf_minibatch = None
        self.use_tf_function = None
        self.override_n_batches = None
        self.index_list = None
        self.sample_configurations = None
        self.key_list = None
        self.rdf = None

    @call
    def __call__(
        self,
        plot: bool = True,
        number_of_bins: int = None,
        cutoff: float = None,
        save: bool = True,
        start: int = 0,
        stop: int = None,
        number_of_configurations: int = 500,
        atom_selection: Union[np.s_, dict] = np.s_[:],
        minibatch: int = -1,
        species: list = None,
        molecules: bool = False,
        **kwargs,
    ):
        """
        Compute the RDF with the given user parameters.

        Parameters
        ----------
        plot: bool
            Plot the RDF after the computation
        number_of_bins: int
            The number of bins for the RDF histogram
        species : list
            A list of species to study.
        cutoff: float
            The cutoff value for the RDF. Default is half the box size
        save: bool
            save the data
        start: int
            Starting position in the database. All values before start will be
            ignored.
        stop: int
            Stopping position in the database. All values after stop will be
            ignored.
        number_of_configurations: int
            The number of uniformly sampled configuration between start and
            stop to be used for the RDF.
        atom_selection : Union[np.s_, dict]
                Atoms to be used in the analysis.
        minibatch: int
            Size of a minibatch over atoms in the batch over configurations.
            Decrease this value if you run into memory
            issues. Increase this value for better performance.
        molecules: bool
            If true, the molecules will be analyzed rather than the atoms.
        kwargs:
            overide_n_batches: int
                    override the automatic batch size calculation
            use_tf_function : bool
                    If true, tf.function is used in the calculation.

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
        )
        # args parsing that will not affect the computation result
        # usually performance or plotting
        self.rdf_minibatch = minibatch
        self.plot = plot

        # kwargs parsing
        self.use_tf_function = kwargs.pop("use_tf_function", False)
        self.override_n_batches = kwargs.get("batches")
        self.tqdm_limit = kwargs.pop("tqdm", 10)

    def check_input(self):
        """
        Check the input of the call method and store defaults if needed.

        Returns
        -------
        Updates class attributes if required.

        """
        if self.args.stop is None:
            self.args.stop = self.experiment.number_of_configurations - 1

        if self.args.cutoff is None:
            self.args.cutoff = (
                self.experiment.box_array[0] / 2 - 0.1
            )  # set cutoff to half box size if none set

        if self.args.number_of_configurations == -1:
            self.args.number_of_configurations = (
                self.experiment.number_of_configurations - 1
            )

        if self.rdf_minibatch == -1:
            self.rdf_minibatch = self.args.number_of_configurations

        if self.args.number_of_bins is None:
            self.args.number_of_bins = int(
                self.args.cutoff / 0.01
            )  # default is 1/100th of an angstrom

        # Get the correct species out.
        if self.args.species is None:
            if self.args.molecules:
                self.args.species = list(self.experiment.molecules)
            else:
                self.args.species = list(self.experiment.species)
        self._initialize_rdf_parameters()

    def _initialize_rdf_parameters(self):
        """
        Initialize the RDF parameters.

        Returns
        -------
        Updates class attributes.

        """
        self.bin_range = [0, self.args.cutoff]
        self.index_list = list(range(len(self.args.species)))
        # Get the indices of the species

        self.sample_configurations = np.linspace(
            self.args.start,
            self.args.stop,
            self.args.number_of_configurations,
            dtype=int,
        )  # choose sampled configurations

        # Generate the tuples e.g ('Na', 'Cl'), ('Na', 'Na')
        self.key_list = [
            self._get_species_names(x)
            for x in list(itertools.combinations_with_replacement(self.index_list, r=2))
        ]

        self.rdf = {
            name: np.zeros(self.args.number_of_bins) for name in self.key_list
        }  # instantiate the rdf tuples

    def _get_species_names(self, species_tuple: tuple) -> str:
        """
        Get the correct names of the species being studied.

        Parameters
        ----------
        species_tuple : tuple
                The species tuple i.e (1, 2) corresponding to the rdf being calculated

        Returns
        -------
        names : str
                Prefix for the saved file

        """
        arg_1 = self.args.species[species_tuple[0]]
        arg_2 = self.args.species[species_tuple[1]]
        return f"{arg_1}_{arg_2}"

    def _calculate_prefactor(self, species: Union[str, tuple] = None):
        """
        Calculate the relevant prefactor for the analysis.

        Parameters
        ----------
        species : str
                The species tuple of the RDF being studied, e.g. Na_Na

        """
        species_scale_factor = 1
        species_split = species.split("_")
        if species_split[0] == species_split[1]:
            species_scale_factor = 2

        if self.args.molecules:
            # Density of all atoms / total volume
            rho = (
                self.experiment.molecules[species_split[1]].n_particles
                / self.experiment.volume
            )
            numerator = species_scale_factor
            denominator = (
                self.args.number_of_configurations
                * rho
                * self.ideal_correction
                * self.experiment.molecules[species_split[0]].n_particles
            )
        else:
            if isinstance(self.args.atom_selection, dict):
                n_species_0 = len(self.args.atom_selection[species_split[0]])
                n_species_1 = len(self.args.atom_selection[species_split[1]])
            else:
                n_species_0 = self.experiment.species[species_split[0]].n_particles
                n_species_1 = self.experiment.species[species_split[1]].n_particles

            # Density of all atoms / total volume
            rho = n_species_1 / self.experiment.volume
            numerator = species_scale_factor
            denominator = (
                self.args.number_of_configurations
                * rho
                * self.ideal_correction
                * n_species_0
            )
        prefactor = numerator / denominator

        return prefactor

    def _calculate_radial_distribution_functions(self):
        """
        Take the calculated histograms and apply the correct pre-factor to them to get
        the correct RDF.

        Returns
        -------
        Updates the class state with the full RDF for each desired species pair.

        """
        # Compute the true RDF for each species combination.
        self.rdf.update(
            {key: np.array(val.numpy(), dtype=float) for key, val in self.rdf.items()}
        )

        for names in self.key_list:
            self.selected_species = names.split("_")
            # TODO use selected_species instead of names, it is more clear
            prefactor = self._calculate_prefactor(names)  # calculate the prefactor

            self.rdf.update(
                {names: self.rdf.get(names) * prefactor}
            )  # Apply the prefactor
            log.debug("Writing RDF to database!")

            x_data = self._ang_to_nm(
                np.linspace(0.0, self.args.cutoff, self.args.number_of_bins)
            )
            y_data = self.rdf.get(names)

            # self.data_range = self.number_of_configurations
            data = {
                self.result_series_keys[0]: x_data.tolist(),
                self.result_series_keys[1]: y_data.tolist(),
            }

            self.queue_data(data=data, subjects=self.selected_species)

    def _ang_to_nm(self, data_in: np.ndarray) -> np.ndarray:
        """
        Convert Angstroms to nm.

        Returns
        -------
        data_out : np.ndarray
                data_in converted to nm

        """
        return (self.experiment.units.length / 1e-9) * data_in

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

    def run_minibatch_loop(self, atoms, stop, n_atoms, minibatch_start, positions_tensor):
        """
        Run a minibatch loop.

        Parameters
        ----------
        atoms : tf.Tensor
        stop : int
        n_atoms : int
        minibatch_start : int
        positions_tensor : tf.Tensor

        """
        # Compute the number of atoms and configurations in the batch.
        atoms_per_batch, batch_size, _ = tf.shape(atoms)

        # Compute the indices
        stop += atoms_per_batch
        start_time = timer()
        indices = get_partial_triu_indices(n_atoms, atoms_per_batch, minibatch_start)
        log.debug(f"Calculating indices took {timer() - start_time} s")

        # Compute the d_ij matrix.
        start_time = timer()
        d_ij = self.get_dij(
            indices,
            positions_tensor,
            atoms,
            tf.cast(self.experiment.box_array, dtype=self.dtype),
        )
        exec_time = timer() - start_time
        atom_pairs_per_second = (
            tf.cast(tf.shape(indices)[1], dtype=self.dtype) / exec_time / 10**6
        )
        atom_pairs_per_second *= tf.cast(batch_size, dtype=self.dtype)
        log.debug(
            f"Computing d_ij took {exec_time} s "
            f"({atom_pairs_per_second:.1f} million atom pairs / s)"
        )

        # Compute the rdf for the minibatch
        start_time = timer()
        minibatch_rdf = self.compute_species_values(indices, minibatch_start, d_ij)
        log.debug(f"Computing species values took {timer() - start_time} s")

        minibatch_start = stop
        return minibatch_rdf, minibatch_start, stop

    def compute_species_values(
        self, indices: tf.Tensor, start_batch, d_ij: tf.Tensor
    ) -> dict:
        """
        Compute species-wise histograms.

        Parameters
        ----------
        indices : tf.Tensor
                indices of the d_ij distances in the shape (x, 2)
        start_batch :
                starts from 0 and increments by atoms_per_batch every batch
        d_ij : tf.Tensor
                d_ij matrix in the shape (x, batches) where x comes from the triu
                computation
        start_batch : int
                Atom index within a single configuration from to start the computation
                based on the current minibatch that is being computed.

        Returns
        -------
        rdf : dict
                Dict of rdf values for each combination of species, e.g.:
                {'H-O': tf.Tensor(...), 'H-H': ..., 'O-O': ...}

        """
        rdf = {
            name: tf.zeros(self.args.number_of_bins, dtype=tf.int32)
            for name in self.key_list
        }
        indices = tf.transpose(indices)

        particles_list = self.particles_list
        for tuples in itertools.combinations_with_replacement(self.index_list, 2):
            names = self._get_species_names(tuples)
            start_ = tf.stack(
                [
                    sum(particles_list[: tuples[0]]) - start_batch,
                    sum(particles_list[: tuples[1]]),
                ],
                axis=0,
            )
            stop_ = start_ + tf.constant(
                [particles_list[tuples[0]], particles_list[tuples[1]]]
            )

            rdf[names] = self.bin_minibatch(
                start_,
                stop_,
                indices,
                d_ij,
                tf.cast(self.bin_range, dtype=self.dtype),
                tf.cast(self.args.number_of_bins, dtype=tf.int32),
                tf.cast(self.args.cutoff, dtype=self.dtype),
            )
        return rdf

    def plot_data(self, data):
        """Plot the RDF data."""
        for selected_species, val in data.items():
            self.run_visualization(
                x_data=np.array(val[self.result_series_keys[0]]),
                y_data=np.array(val[self.result_series_keys[1]]),
                title=selected_species,
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
        data : tf.Tensor
                data tensor of the shape (n_atoms * n_species, n_configurations, 3)

        """
        formatted_data = []
        for item in keys:
            formatted_data.append(batch[item])

        if len(self.args.species) == 1:
            return tf.cast(formatted_data[0], dtype=self.dtype)
        else:
            return tf.cast(tf.concat(formatted_data, axis=0), dtype=self.dtype)

    def prepare_computation(self):
        """
        Run the steps necessary to prepare for the RDF computation.

        Returns
        -------
        dict_keys : list
                dict keys to use when selecting data from the output.
        split_arr : np.ndarray
                Array of indices to load from the database split into sub-arrays which
                fulfill the necessary batch size.
        batch_tqdm : bool
                If true, the main tqdm loop over batches is disabled and only the
                mini-batch loop will be displayed.

        """
        path_list = [
            join_path(item, self.loaded_property.name) for item in self.args.species
        ]
        self._prepare_managers(path_list)

        # batch loop correction
        self._correct_batch_properties()

        # Get the correct dict keys.
        dict_keys = []
        for item in self.args.species:
            dict_keys.append(str.encode(join_path(item, self.loaded_property.name)))

        # Split the configurations into batches.
        split_arr = np.array_split(self.sample_configurations, self.n_batches)

        # Turn off the tqdm for certain scenarios.
        batch_tqdm = self.tqdm_limit > self.n_batches

        return dict_keys, split_arr, batch_tqdm

    @staticmethod
    def combine_dictionaries(dict_a: dict, dict_b: dict):
        """
        Combine two dictionaries in a tf.function call.

        Parameters
        ----------
        dict_a : dict
        dict_b : dict

        """
        out = {}
        for key in dict_a:
            out[key] = dict_a[key] + dict_b[key]
        return out

    @staticmethod
    @tf.function(experimental_relax_shapes=True)
    def bin_minibatch(
        start, stop, indices, d_ij, bin_range, number_of_bins, cutoff
    ) -> tf.Tensor:
        """
        Compute the minibatch histogram.

        Parameters
        ----------
        start : list
        stop : list
        indices : tf.Tensor
        d_ij : tf.Tensor
        bin_range
        number_of_bins : int
        cutoff : float
                Cutoff to enforce on the distance tensor.

        """
        # select the indices that are within the boundaries of the current species /
        # molecule
        mask_1 = (indices[:, 0] > start[0]) & (indices[:, 0] < stop[0])
        mask_2 = (indices[:, 1] > start[1]) & (indices[:, 1] < stop[1])

        values_species = tf.boolean_mask(d_ij, mask_1 & mask_2, axis=0)
        values = apply_system_cutoff(values_species, cutoff)
        bin_data = tf.histogram_fixed_width(
            values=values, value_range=bin_range, nbins=number_of_bins
        )
        return bin_data

    @staticmethod
    @tf.function(experimental_relax_shapes=True)
    def get_dij(indices, positions_tensor, atoms, box_array):
        """
        Compute the distance matrix for the minibatch.

        Parameters
        ----------
        indices : tf.Tensor
        positions_tensor : tf.Tensor
        atoms : tf.Tensor
        box_array : tf.Tensor

        """
        start_time = timer()
        log.debug(f"Calculating indices took {timer() - start_time} s")

        # apply the mask to this, to only get the triu values and don't compute
        # anything twice
        start_time = timer()
        _positions = tf.gather(positions_tensor, indices[1], axis=0)
        log.debug(f"Gathering positions_tensor took {timer() - start_time} s")

        # for atoms_per_batch > 1, flatten the array according to the positions
        start_time = timer()
        atoms_position = tf.gather(atoms, indices[0], axis=0)
        log.debug(f"Gathering atoms took {timer() - start_time} s")

        start_time = timer()
        r_ij = _positions - atoms_position
        log.debug(f"Computing r_ij took {timer() - start_time} s")

        # apply minimum image convention
        start_time = timer()
        if box_array is not None:
            r_ij = apply_minimum_image(r_ij, box_array)
        log.debug(f"Applying minimum image convention took {timer() - start_time} s")

        start_time = timer()
        d_ij = tf.linalg.norm(r_ij, axis=-1)
        log.debug(f"Computing d_ij took {timer() - start_time} s")

        return d_ij

    @property
    def particles_list(self):
        """
        List of number of atoms of each species being studied.

        Returns
        -------
        -------.

        """
        if self.args.molecules:
            particles_list = [
                self.experiment.molecules[item].n_particles
                for item in self.experiment.molecules
            ]
        else:
            if isinstance(self.args.atom_selection, dict):
                particles_list = [
                    len(self.args.atom_selection[item]) for item in self.args.species
                ]
            else:
                particles_list = [
                    self.experiment.species[item].n_particles
                    for item in self.args.species
                ]

        return particles_list

    @property
    def ideal_correction(self) -> float:
        """Get the correct ideal gas term.

        In the case of a cutoff value greater than half of the box size, the ideal gas
        term of the experiment must be corrected due to the lack of spherical symmetry
        in the experiment.

        Returns
        -------
        correction : float
                Correct ideal gas term for the RDF prefactor

        """

        def _spherical_symmetry(data: np.array) -> np.array:  # TODO make it a property
            """
            Operation to perform for full spherical symmetry.

            Parameters
            ----------
            data : np.array
                    tensor_values on which to operate

            Returns
            -------
            function_values : np.array
                    result of the operation

            """
            return 4 * np.pi * (data**2)

        def _correction_1(data: np.array) -> np.array:
            """
            First correction to ideal gas.

            tensor_values : np.array
                    tensor_values on which to operate

            Returns
            -------
            function_values : np.array
                    result of the operation

            """
            return 2 * np.pi * data * (3 - 4 * data)

        def _correction_2(data: np.array) -> np.array:
            """
            Second correction to ideal gas.

            tensor_values : np.array
                    tensor_values on which to operate

            Returns
            -------
            function_values : np.array
                    result of the operation

            """
            arctan_1 = np.arctan(np.sqrt(4 * (data**2) - 2))
            arctan_2 = (
                8
                * data
                * np.arctan(
                    (2 * data * (4 * (data**2) - 3))
                    / (np.sqrt(4 * (data**2) - 2) * (4 * (data**2) + 1))
                )
            )

            return 2 * data * (3 * np.pi - 12 * arctan_1 + arctan_2)

        def _piecewise(data: np.array) -> np.array:
            """
            Return a piecewise operation on a set of tensor_values

            Parameters
            ----------
            data : np.array
                    tensor_values on which to operate.

            Returns
            -------
            scaled_data : np.array
                    tensor_values that has been operated on.

            """
            # Boundaries on the ideal gsa correction. These go to 73% over half the box
            # size, the most for a cubic box.
            lower_bound = self.experiment.box_array[0] / 2
            middle_bound = np.sqrt(2) * self.experiment.box_array[0] / 2

            # split the tensor_values into parts
            split_1 = list(split_array(data, data <= lower_bound))
            if len(split_1) == 1:
                return _spherical_symmetry(split_1[0])
            else:
                split_2 = list(split_array(split_1[1], split_1[1] < middle_bound))
                if len(split_2) == 1:
                    return np.concatenate(
                        (_spherical_symmetry(split_1[0]), _correction_1(split_2[0]))
                    )
                else:
                    return np.concatenate(
                        (
                            _spherical_symmetry(split_1[0]),
                            _correction_1(split_2[0]),
                            _correction_2(split_2[1]),
                        )
                    )

        bin_width = self.args.cutoff / self.args.number_of_bins
        bin_edges = np.linspace(0.0, self.args.cutoff, self.args.number_of_bins)

        return _piecewise(np.array(bin_edges)) * bin_width

    def run_calculator(self):
        """
        Run the analysis.

        Returns
        -------

        """
        self.check_input()

        dict_keys, split_arr, batch_tqm = self.prepare_computation()

        # Get the batch dataset
        batch_ds = self.get_batch_dataset(
            subject_list=self.args.species, loop_array=split_arr, correct=True
        )

        # Loop over the batches.
        for idx, batch in tqdm(
            enumerate(batch_ds), ncols=70, disable=batch_tqm, total=self.n_batches
        ):
            # Reformat the data.
            log.debug("Reformatting data.")
            positions_tensor = self._format_data(batch=batch, keys=dict_keys)

            # Create a new dataset to loop over.
            log.debug("Creating dataset.")
            per_atoms_ds = tf.data.Dataset.from_tensor_slices(positions_tensor)
            n_atoms = tf.shape(positions_tensor)[0]

            # Start the computation.
            log.debug("Beginning calculation.")
            minibatch_start = tf.constant(0)
            stop = tf.constant(0)
            rdf = {
                name: tf.zeros(self.args.number_of_bins, dtype=tf.int32)
                for name in self.key_list
            }

            for atoms in tqdm(
                per_atoms_ds.batch(self.rdf_minibatch).prefetch(tf.data.AUTOTUNE),
                ncols=70,
                disable=not batch_tqm,
                desc=f"Running mini batch loop {idx + 1} / {self.n_batches}",
            ):
                # Compute the minibatch update
                minibatch_rdf, minibatch_start, stop = self.run_minibatch_loop(
                    atoms, stop, n_atoms, minibatch_start, positions_tensor
                )

                # Update the rdf.
                start_time = timer()
                rdf = self.combine_dictionaries(rdf, minibatch_rdf)
                log.debug(f"Updating dictionaries took {timer() - start_time} s")

            # Update the class before the next batch.
            for key in self.rdf:
                self.rdf[key] += rdf[key]

        self._calculate_radial_distribution_functions()
