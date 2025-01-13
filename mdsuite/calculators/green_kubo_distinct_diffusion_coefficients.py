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
Module for computing distinct diffusion coefficients using the Green-Kubo method.
"""

import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any, List, Union

import jax
import numpy as np
import tensorflow as tf
from bokeh.models import Span
from tqdm import tqdm

from mdsuite.calculators.calculator import call
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.utils.calculator_helper_methods import correlate


@dataclass
class Args:
    """Data class for the saved properties."""

    data_range: int
    correlation_time: int
    atom_selection: np.s_
    tau_values: np.s_
    molecules: bool
    species: list
    integration_range: int


class GreenKuboDistinctDiffusionCoefficients(TrajectoryCalculator, ABC):
    """
    Class for the Green-Kubo diffusion coefficient implementation

    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    x_label : str
            X label of the tensor_values when plotted
    y_label : str
            Y label of the tensor_values when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database_path for the analysis.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.GreenKuboDistinctDiffusionCoefficients(data_range=500,
    plot=True, correlation_time=10)

    """

    def __init__(self, **kwargs):
        """
        Constructor for the Green Kubo diffusion coefficients class.

        Attributes
        ----------
        experiment :  object
                Experiment class to call from

        """
        super().__init__(**kwargs)

        self.scale_function = {"quadratic": {"inner_scale_factor": 5}}
        self.loaded_property = mdsuite_properties.velocities

        self.database_group = "Diffusion_Coefficients"
        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"$$\text{VACF}  / m^{2}/s^{2}$$"
        self.analysis_name = "Green_Kubo_Distinct_Diffusion_Coefficients"
        self.experimental = True
        self.result_keys = ["diffusion_coefficient", "uncertainty"]
        self.result_series_keys = ["time", "vacf"]
        self._dtype = tf.float64
        self.sigma = []

    @call
    def __call__(
        self,
        plot: bool = False,
        species: list = None,
        data_range: int = 500,
        save: bool = True,
        correlation_time: int = 1,
        tau_values: Union[int, List, Any] = np.s_[:],
        molecules: bool = False,
        export: bool = False,
        atom_selection: dict = np.s_[:],
        integration_range: int = None,
    ):
        """
        Constructor for the Green Kubo diffusion coefficients class.

        Parameters
        ----------
        plot : bool
                if true, plot the output.
        species : list
                List of species on which to operate.
        data_range : int
                Data range to use in the analysis.
        save : bool
                if true, save the output.
        correlation_time : int
                Correlation time to use in the window sampling.
        molecules : bool
                If true, molecules are used instead of atoms.
        atom_selection : np.s_
                Selection of atoms to use within the HDF5 database.
        export : bool
                If true, export the data directly into a csv file.
        integration_range : int
                Range over which to perform the integration.

        """
        if integration_range is None:
            integration_range = data_range

        # set args that will affect the computation result
        self.args = Args(
            data_range=data_range,
            correlation_time=correlation_time,
            atom_selection=atom_selection,
            tau_values=tau_values,
            molecules=molecules,
            species=species,
            integration_range=integration_range,
        )

        self.plot = plot
        self.time = self._handle_tau_values()

        self.species = species  # Which species to calculate for

        self.vacf = np.zeros(self.args.data_range)

        if self.species is None:
            self.species = list(self.experiment.species)

        self.combinations = list(itertools.combinations_with_replacement(self.species, 2))

    def _compute_self_correlation(self, ds_a, ds_b):
        """
        Compute the self correlation coefficients.

        Parameters
        ----------
        ds_a : np.ndarray (n_timesteps, n_atoms, dimension)
        ds_b : np.ndarray (n_timesteps, n_atoms, dimension)
        data_range : int (default = 500)
                Range over which the acf will be computed.
        correlation_time : int (default = 1).

        Returns
        -------
        -------.

        """
        atomwise_vmap = jax.vmap(correlate, in_axes=0)

        return np.mean(atomwise_vmap(ds_a, ds_b), axis=0)

    def _map_over_particles(self, ds_a: np.ndarray, ds_b: np.ndarray) -> np.ndarray:
        """
        Function to map a correlation in a Gram matrix style over two data sets.

        This function will perform the nxm calculations to compute the correlation
        between all particles in ds_a with all particles in ds_b.

        Parameters
        ----------
        ds_a : np.ndarray (n_particles, n_configurations, dimension)
                Dataset to compute correlation with.
        ds_b : np.ndarray (n_particles, n_configurations, dimension)
                Other dataset to compute correlation with. Does not need to be the
                same shape as ds_a along the zeroth (particle) axis.

        Returns
        -------

        """

        def ref_conf_map(ref_dataset, full_ds):
            """
            Maps over the atoms axis in dataset

            Parameters
            ----------
            dataset
            Returns
            -------.

            """

            def test_conf_map(test_dataset):
                """
                Map over atoms in test dataset.

                Parameters
                ----------
                test_dataset
                Returns
                -------.

                """
                return correlate(ref_dataset, test_dataset)

            return np.mean(jax.vmap(test_conf_map, in_axes=0)(full_ds), axis=0)

        acf_calc = jax.vmap(ref_conf_map, in_axes=(0, None))

        return np.mean(acf_calc(ds_a, ds_b), axis=0)

    def ensemble_operation(self, data: dict, dict_ref: list, same_species: bool = False):
        """
        Compute the vacf on the given dictionary of data.

        Parameters
        ----------
        dict_ref : tuple:
                Names of the entries in the dictionary. Used to select a specific
                element.
        data : dict
                Dictionary of data returned by tensorflow
        same_species : bool
                If true, the species are the same and i=j should be skipped.

        Returns
        -------
        updates the class state

        """
        vacf = self._map_over_particles(
            data[dict_ref[0]].numpy(), data[dict_ref[1]].numpy()
        )
        if same_species:
            self_correlation = self._compute_self_correlation(
                data[dict_ref[0]].numpy(), data[dict_ref[1]].numpy()
            )
            vacf -= self_correlation
        self.vacf += vacf
        self.sigma.append(np.trapz(vacf, x=self.time))

    def run_calculator(self):
        """Perform the distinct coefficient analysis analysis."""
        self.check_input()
        for combination in self.combinations:
            species_values = list(combination)
            dict_ref = [
                str.encode("/".join([species, self.loaded_property.name]))
                for species in species_values
            ]
            batch_ds = self.get_batch_dataset(species_values)

            for batch in tqdm(
                batch_ds,
                ncols=70,
                desc=f"{combination[0]}-{combination[1]}",
                total=self.n_batches,
                disable=self.memory_manager.minibatch,
            ):
                ensemble_ds = self.get_ensemble_dataset(batch, species_values)
                for ensemble in ensemble_ds:
                    self.ensemble_operation(
                        ensemble, dict_ref, species_values[0] == species_values[1]
                    )

            self._calculate_prefactor(combination)
            self._post_operation_processes(combination)
            self.sigma = []

    def _calculate_prefactor(self, species: Union[str, tuple] = None):
        """
        Calculate the calculator pre-factor.

        Parameters
        ----------
        species : str
                Species property if required.

        Returns
        -------

        """
        numerator = self.experiment.units.length**2
        denominator = self.experiment.units.time * (self.args.data_range - 1)

        self.prefactor = numerator / denominator

    def check_input(self):
        """
        Check the user input to ensure no conflicts are present.

        Returns
        -------

        """
        self._run_dependency_check()

    def _post_operation_processes(self, species: Union[str, tuple] = None):
        """
        Call the post-op processes
        Returns
        -------.

        """
        result = self.prefactor * np.array(self.sigma)

        data = {
            self.result_keys[0]: np.mean(result).tolist(),
            self.result_keys[1]: (np.std(result) / (np.sqrt(len(result)))).tolist(),
            self.result_series_keys[0]: self.time.tolist(),
            self.result_series_keys[1]: self.vacf.tolist(),
        }

        self.queue_data(data=data, subjects=list(species))

    def plot_data(self, data):
        """Plot the data."""
        for selected_species, val in data.items():
            span = Span(
                location=(
                    np.array(val[self.result_series_keys[0]]) * self.experiment.units.time
                )[self.args.integration_range - 1],
                dimension="height",
                line_dash="dashed",
            )
            self.run_visualization(
                x_data=np.array(val[self.result_series_keys[0]])
                * self.experiment.units.time,
                y_data=np.array(val[self.result_series_keys[1]]),
                title=(
                    f"{selected_species}: {val[self.result_keys[0]]: 0.3E} +-"
                    f" {val[self.result_keys[1]]: 0.3E}"
                ),
                layouts=[span],
            )
