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
MDSuite module for the computation of viscosity using the Green-Kubo relation as applied
to the stress on a system.
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


class GreenKuboViscosityFlux(TrajectoryCalculator, ABC):
    """
    Class for the Green Kubo viscosity from flux implementation.

    Attributes
    ----------
    experiment :  object
            Experiment class to call from

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.GreenKuboViscosityFlux(data_range=500,
                                                      plot=True,
                                                      correlation_time=10)

    """

    def __init__(self, **kwargs):
        """
        Python constructor for the experiment class.

        Parameters
        ----------
        experiment : object
                Experiment class to read and write to

        """
        super().__init__(**kwargs)
        self.scale_function = {"linear": {"scale_factor": 5}}

        self.loaded_property = mdsuite_properties.stress_viscosity
        self.system_property = True

        self.analysis_name = "Viscosity_Flux"
        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"$$\text{JACF} / C^{2}\\cdot m^{2}/s^{2}$$"

        self.prefactor: float
        self.jacf: np.ndarray
        self.sigma = []
        self._dtype = tf.float64

    @call
    def __call__(
        self,
        plot=False,
        data_range=500,
        correlation_time=1,
        tau_values: np.s_ = np.s_[:],
        integration_range: int = None,
    ):
        """
        Python constructor for the experiment class.

        Parameters
        ----------
        plot : bool
                If true, a plot of the analysis is saved.
        data_range : int
                Number of configurations to include in each ensemble

        """
        self.plot = plot
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
        Compute the ionic conductivity prefactor.

        Returns
        -------

        """
        # Calculate the prefactor
        numerator = self.experiment.volume
        denominator = (
            3
            * (self.args.data_range - 1)
            * self.experiment.temperature
            * self.experiment.units.boltzmann
        )

        prefactor_units = (
            self.experiment.units.pressure**2
            * self.experiment.units.volume
            * self.experiment.units.time
            / self.experiment.units.energy
        )

        self.prefactor = (numerator / denominator) * prefactor_units

    def _apply_averaging_factor(self):
        """
        Apply the averaging factor to the msd array.

        Returns
        -------
        -------.

        """
        self.jacf /= max(self.jacf)

    def ensemble_operation(self, ensemble):
        """
        Calculate and return the vacf.

        Parameters
        ----------
        ensemble

        Returns
        -------
        updates class vacf with the tensor_values.

        """
        jacf = self.args.data_range * tf.reduce_sum(
            tfp.stats.auto_correlation(ensemble, normalize=False, axis=0, center=False),
            axis=-1,
        )
        self.jacf += jacf[int(self.args.data_range - 1) :]
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
            "viscosity": result[0],
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
