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
Module for computing distinct diffusion coefficients using the Einstein method.
"""

import itertools
import warnings
from dataclasses import dataclass
from typing import Any, List, Union

import jax
import numpy as np
from tqdm import tqdm

from mdsuite.calculators.calculator import call
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.utils.calculator_helper_methods import fit_einstein_curve, msd_operation


@dataclass
class Args:
    """Data class for the saved properties."""

    data_range: int
    correlation_time: int
    atom_selection: np.s_
    tau_values: np.s_
    molecules: bool
    species: list
    fit_range: int


tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class EinsteinDistinctDiffusionCoefficients(TrajectoryCalculator):
    """
    Class for the Green-Kubo diffusion coefficient implementation.

    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    species : list
            Which species to perform the analysis on
    x_label : str
            X label of the tensor_values when plotted
    y_label : str
            Y label of the tensor_values when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database_path for the analysis

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.EinsteinDistinctDiffusionCoefficients(data_range=500,
                                                                     plot=True,
                                                                     correlation_time=10)

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

        self.scale_function = {"quadratic": {"inner_scale_factor": 10}}
        self.loaded_property = mdsuite_properties.unwrapped_positions

        self.database_group = "Diffusion_Coefficients"
        self.x_label = r"$$\text{Time} / s $$"
        self.y_label = r"$$\text{MSD} / m^{2}$$"
        self.analysis_name = "Einstein_Distinct_Diffusion_Coefficients"
        self.experimental = True
        self.result_keys = ["diffusion_coefficient", "uncertainty"]
        self.result_series_keys = ["time", "msd"]
        self.combinations = []

    @call
    def __call__(
        self,
        plot: bool = True,
        species: list = None,
        data_range: int = 500,
        save: bool = True,
        correlation_time: int = 1,
        tau_values: Union[int, List, Any] = np.s_[:],
        molecules: bool = False,
        export: bool = False,
        atom_selection: dict = np.s_[:],
        fit_range: int = -1,
    ):
        """
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
        atom_selection : np.s_
                Selection of atoms to use within the HDF5 database.
        export : bool
                If true, export the data directly into a csv file.

        Returns
        -------
        None

        """
        if species is None:
            species = list(self.experiment.species)
        self.combinations = list(itertools.combinations_with_replacement(species, 2))

        self.plot = plot

        if fit_range == -1:
            fit_range = int(data_range - 1)

        # set args that will affect the computation result
        self.args = Args(
            data_range=data_range,
            correlation_time=correlation_time,
            atom_selection=atom_selection,
            tau_values=tau_values,
            molecules=molecules,
            species=species,
            fit_range=fit_range,
        )
        self.time = self._handle_tau_values() * self.experiment.units.time

        self.msd_array = np.zeros(self.args.data_range)  # define empty msd array

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
            ----------.

            Returns
            -------

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
                return msd_operation(ref_dataset, test_dataset)

            return np.mean(jax.vmap(test_conf_map, in_axes=0)(full_ds), axis=0)

        acf_calc = jax.vmap(ref_conf_map, in_axes=(0, None))

        return np.mean(acf_calc(ds_a, ds_b), axis=0)

    def _compute_self_correlation(self, ds_a, ds_b):
        """
        Compute the self correlation coefficients.

        Parameters
        ----------
        ds_a : np.ndarray (n_timesteps, n_atoms, dimension)
        ds_b : np.ndarray (n_timesteps, n_atoms, dimension).

        Returns
        -------

        """
        atomwise_vmap = jax.vmap(msd_operation, in_axes=0)

        return np.mean(atomwise_vmap(ds_a, ds_b), axis=0)

    def _compute_msd(self, data: dict, data_path: list, combination: tuple):
        """
        Compute the msd on the given dictionary of data.

        Parameters
        ----------
        data : dict
                Dictionary of data returned by tensorflow.
        data_path : list
                Data paths for accessing the dictionary.
        combination : tuple
                Tuple being studied in the msd, i.e. ('Na', 'Cl) or ('Na', 'Na').

        Returns
        -------
        updates the class state

        """
        msd_array = self._map_over_particles(
            data[data_path[0]].numpy(), data[data_path[1]].numpy()
        )

        if combination[0] == combination[1]:
            self_correction = self._compute_self_correlation(
                data[data_path[0]].numpy(), data[data_path[1]].numpy()
            )
            msd_array -= self_correction

        self.msd_array += msd_array

    def _apply_averaging_factor(self):
        """
        Apply an averaging factor to the tensor_values.

        Returns
        -------
        averaged copy of the tensor_values.

        """
        self.msd_array /= int(self.n_batches) * self.ensemble_loop

    def _post_operation_processes(self, species: Union[str, tuple] = None):
        """
        Call the post-op processes
        Returns
        -------.

        """
        self._apply_averaging_factor()  # update in place
        self.msd_array *= self.experiment.units.length**2
        try:
            fit_values, covariance, gradients, gradient_errors = fit_einstein_curve(
                x_data=self.time, y_data=self.msd_array, fit_max_index=self.args.fit_range
            )
            error = np.sqrt(np.diag(covariance))[0]

            data = {
                self.result_keys[0]: 1 / 2 * fit_values[0],
                self.result_keys[1]: 1 / 2 * error,
                self.result_series_keys[0]: self.time.tolist(),
                self.result_series_keys[1]: self.msd_array.tolist(),
            }

        except ValueError:
            fit_values, covariance, gradients, gradient_errors = fit_einstein_curve(
                x_data=self.time,
                y_data=abs(self.msd_array),
                fit_max_index=self.args.fit_range,
            )
            error = np.sqrt(np.diag(covariance))[0]
            # division by dimension is performed in the mapping, therefore, only 2 here.
            data = {
                self.result_keys[0]: -1 / 2 * fit_values[0],
                self.result_keys[1]: 1 / 2 * error,
                self.result_series_keys[0]: self.time.tolist(),
                self.result_series_keys[1]: self.msd_array.tolist(),
            }

        data.update({"time": self.time.tolist(), "msd": self.msd_array.tolist()})

        self.queue_data(data=data, subjects=list(species))

    def check_input(self):
        """
        Check the user input to ensure no conflicts are present.

        Returns
        -------

        """
        self._run_dependency_check()

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
                    self._compute_msd(ensemble, dict_ref, combination)

            self._post_operation_processes(combination)
            self.msd_array = np.zeros(self.args.data_range)  # define empty msd array
