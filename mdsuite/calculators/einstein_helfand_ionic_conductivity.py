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
MDSuite module for the computation of ionic conductivity using the Einstein method.
"""

from abc import ABC
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.calculators.calculator import call
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.utils import DatasetKeys
from mdsuite.utils.calculator_helper_methods import fit_einstein_curve
from mdsuite.utils.units import boltzmann_constant, elementary_charge


@dataclass
class Args:
    """Data class for the saved properties."""

    data_range: int
    correlation_time: int
    tau_values: np.s_
    atom_selection: np.s_
    fit_range: int


class EinsteinHelfandIonicConductivity(TrajectoryCalculator, ABC):
    """
    Class for the Einstein-Helfand Ionic Conductivity.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.EinsteinHelfandTIonicConductivity(data_range=500,
                                                                 plot=True,
                                                                 correlation_time=10)

    """

    def __init__(self, **kwargs):
        """
        Python constructor.

        Parameters
        ----------
        experiment :  object
            Experiment class to call from

        """
        # parse to the experiment class
        super().__init__(**kwargs)
        self.scale_function = {"linear": {"scale_factor": 5}}

        self.loaded_property = mdsuite_properties.translational_dipole_moment
        self.dependency = mdsuite_properties.unwrapped_positions
        self.system_property = True

        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"$$\text{MSD} / m^2/s$$"
        self.analysis_name = "Einstein Helfand Ionic Conductivity"

        self.result_keys = ["ionic_conductivity", "uncertainty"]
        self.result_series_keys = ["time", "msd"]

        self._dtype = tf.float64

    @call
    def __call__(
        self,
        plot=True,
        data_range=100,
        correlation_time=1,
        tau_values: np.s_ = np.s_[:],
        fit_range: int = -1,
    ):
        """
        Python constructor.

        Parameters
        ----------
        plot : bool
                if true, plot the tensor_values
        data_range :
                Number of configurations to use in each ensemble
        correlation_time : int
                Correlation time to use in the analysis.

        """
        if fit_range == -1:
            fit_range = int(data_range - 1)

        # set args that will affect the computation result
        self.args = Args(
            data_range=data_range,
            correlation_time=correlation_time,
            tau_values=tau_values,
            atom_selection=np.s_[:],
            fit_range=fit_range,
        )

        self.plot = plot
        self.time = self._handle_tau_values()
        self.msd_array = np.zeros(self.data_resolution)

    def check_input(self):
        """
        Check the user input to ensure no conflicts are present.

        Returns
        -------

        """
        self._run_dependency_check()

    def _calculate_prefactor(self):
        """
        Compute the ionic conductivity prefactor.

        Returns
        -------

        """
        # Calculate the prefactor
        numerator = (self.experiment.units.length**2) * (elementary_charge**2)
        denominator = (
            self.experiment.units.time
            * self.experiment.volume
            * self.experiment.units.volume
            * self.experiment.temperature
            * boltzmann_constant
        )
        self.prefactor = numerator / denominator

    def _apply_averaging_factor(self):
        """Apply the averaging factor to the msd array."""
        self.msd_array /= int(self.n_batches) * self.ensemble_loop

    def ensemble_operation(self, ensemble: tf.Tensor):
        """
        Calculate and return the msd.

        Parameters
        ----------
        ensemble

        Returns
        -------
        MSD of the tensor_values.

        """
        msd = tf.math.squared_difference(
            tf.gather(ensemble, self.args.tau_values, axis=1), ensemble[:, 0, :]
        )
        msd = self.prefactor * tf.reduce_sum(msd, axis=2)
        self.msd_array += np.array(msd)[0, :]

    def _post_operation_processes(self):
        """
        Call the post-op processes
        Returns
        -------.

        """
        fit_values, covariance, gradients, gradient_errors = fit_einstein_curve(
            x_data=self.time, y_data=self.msd_array, fit_max_index=self.args.fit_range
        )
        error = np.sqrt(np.diag(covariance))[0]

        data = {
            self.result_keys[0]: 1 / 6 * fit_values[0],
            self.result_keys[1]: 1 / 6 * error,
            self.result_series_keys[0]: self.time.tolist(),
            self.result_series_keys[1]: self.msd_array.tolist(),
        }

        self.queue_data(data=data, subjects=["System"])

    def run_calculator(self):
        """

        Run analysis.

        Returns
        -------

        """
        self.check_input()
        # Compute the pre-factor early.
        self._calculate_prefactor()

        dict_ref = str.encode(
            "/".join([DatasetKeys.OBSERVABLES, self.loaded_property.name])
        )

        batch_ds = self.get_batch_dataset([DatasetKeys.OBSERVABLES])

        for batch in tqdm(
            batch_ds,
            ncols=70,
            total=self.n_batches,
            disable=self.memory_manager.minibatch,
        ):
            ensemble_ds = self.get_ensemble_dataset(batch, DatasetKeys.OBSERVABLES)

            for ensemble in ensemble_ds:
                self.ensemble_operation(ensemble[dict_ref])

        # Scale, save, and plot the data.
        self._apply_averaging_factor()
        self._post_operation_processes()
