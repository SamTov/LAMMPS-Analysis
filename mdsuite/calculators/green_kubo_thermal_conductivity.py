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
MDSuite module for the computation of the thermal conductivity using the Green-Kubo
relation.
"""

from abc import ABC
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from bokeh.models import Span
from tqdm import tqdm

from mdsuite.calculators.calculator import call
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.utils import DatasetKeys


@dataclass
class Args:
    """Data class for the saved properties."""

    data_range: int
    correlation_time: int
    tau_values: np.s_
    atom_selection: np.s_
    integration_range: int


class GreenKuboThermalConductivity(TrajectoryCalculator, ABC):
    """
    Class for the Green-Kubo Thermal conductivity implementation.

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

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.GreenKuboThermalConductivity(data_range=500,
    plot=True, correlation_time=10)

    """

    def __init__(self, **kwargs):
        """
        Class for the Green-Kubo Thermal conductivity implementation.

        Attributes
        ----------
        experiment :  object
                Experiment class to call from

        """
        super().__init__(**kwargs)
        self.scale_function = {"linear": {"scale_factor": 5}}

        self.loaded_property = mdsuite_properties.thermal_flux
        self.system_property = True

        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"$$\text{JACF} / ($C^{2}\cdot m^{2}/s^{2}$$"
        self.analysis_name = "Green_Kubo_Thermal_Conductivity"
        self._dtype = tf.float64

    @call
    def __call__(
        self,
        plot=False,
        data_range=500,
        tau_values: np.s_ = np.s_[:],
        correlation_time: int = 1,
        integration_range: int = None,
    ):
        """
        Class for the Green-Kubo Thermal conductivity implementation.

        Parameters
        ----------
        plot : bool
                if true, plot the output.
        data_range : int
                Data range to use in the analysis.
        correlation_time : int
                Correlation time to use in the window sampling.
        integration_range : int
                Range over which the integration should be performed.

        """
        self.plot = plot
        self.jacf: np.ndarray
        self.prefactor: float
        self.sigma = []

        if integration_range is None:
            integration_range = data_range

        # set args that will affect the computation result
        self.args = Args(
            data_range=data_range,
            correlation_time=correlation_time,
            tau_values=tau_values,
            atom_selection=np.s_[:],
            integration_range=integration_range,
        )

        self.time = self._handle_tau_values()
        self.jacf = np.zeros(self.data_resolution)

    def check_input(self):
        """
        Check the user input to ensure no conflicts are present.

        Returns
        -------

        """
        self._run_dependency_check()

    def _calculate_prefactor(self):
        """
        Compute the ionic conductivity pre-factor.

        Returns
        -------

        """
        # Calculate the prefactor
        # prepare the prefactor for the integral
        numerator = 1
        denominator = (
            3
            * (self.args.data_range - 1)
            * self.experiment.temperature**2
            * self.experiment.units.boltzmann
            * self.experiment.volume
        )
        prefactor_units = (
            self.experiment.units.energy
            / self.experiment.units.length
            / self.experiment.units.time
        )

        self.prefactor = (numerator / denominator) * prefactor_units

    def _apply_averaging_factor(self):
        """
        Apply the averaging factor to the msd array.

        Returns
        -------
        -------.

        """
        pass

    def ensemble_operation(self, ensemble: tf.Tensor):
        """
        Calculate and return the msd.

        Parameters
        ----------
        ensemble : tf.Tenor
                Ensemble to analyze.

        Returns
        -------
        MSD of the tensor_values.

        """
        jacf = self.args.data_range * tf.reduce_sum(
            tfp.stats.auto_correlation(ensemble, normalize=False, axis=0, center=False),
            axis=-1,
        )
        self.jacf += jacf
        self.sigma.append(
            np.trapz(
                jacf[: self.args.integration_range],
                x=self.time[: self.args.integration_range],
            )
        )

    def _post_operation_processes(self):
        """
        Call the post-op processes.

        Returns
        -------

        """
        result = self.prefactor * np.array(self.sigma)

        data = {
            "computation_results": result[0],
            "uncertainty": result[1],
            "time": self.time.tolist(),
            "acf": self.jacf.numpy().tolist(),
        }

        self.queue_data(data=data, subjects=["System"])

        # Update the plot if required
        if self.plot:
            span = Span(
                location=(np.array(self.time) * self.experiment.units.time)[
                    self.args.integration_range - 1
                ],
                dimension="height",
                line_dash="dashed",
            )
            self.run_visualization(
                x_data=np.array(self.time) * self.experiment.units.time,
                y_data=self.jacf.numpy(),
                title=f"{result[0]} +- {result[1]}",
                layouts=[span],
            )

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
