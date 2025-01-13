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
MDSuite module for the computation of the ionic conductivity of a system using the
Green-Kubo relation. Ionic conductivity describes how well a system can conduct an
electrical charge due to the mobility of the ions contained within it. This differs
from electronic conductivity which is transferred by electrons.
"""

from abc import ABC
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from bokeh.models import HoverTool, LinearAxis, Span
from bokeh.models.ranges import Range1d
from bokeh.plotting import figure
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm

from mdsuite.calculators.calculator import call
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.utils import DatasetKeys
from mdsuite.utils.units import boltzmann_constant, elementary_charge


@dataclass
class Args:
    """Data class for the saved properties."""

    data_range: int
    correlation_time: int
    tau_values: np.s_
    atom_selection: np.s_
    integration_range: int


class GreenKuboIonicConductivity(TrajectoryCalculator, ABC):
    """
    Class for the Green-Kubo ionic conductivity implementation.

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
            Property loaded from the database_path for the analysis

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.GreenKuboIonicConductivity(data_range=500,
    plot=True, correlation_time=10)

    """

    def __init__(self, **kwargs):
        """

        Attributes
        ----------
        experiment :  object
                Experiment class to call from

        """
        # update experiment class
        super().__init__(**kwargs)
        self.scale_function = {"linear": {"scale_factor": 5}}

        self.loaded_property = mdsuite_properties.ionic_current
        self.system_property = True

        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"$$\text{JACF} / C^{2}\cdot m^{2}/s^{2}$$"
        self.analysis_name = "Green_Kubo_Ionic_Conductivity"

        self.result_keys = ["ionic_conductivity", "uncertainty"]
        self.result_series_keys = ["time", "acf", "integral", "integral_uncertainty"]

        self.prefactor = None
        self._dtype = tf.float64

    @call
    def __call__(
        self,
        plot=True,
        data_range=500,
        correlation_time=1,
        tau_values: np.s_ = np.s_[:],
        integration_range: int = None,
    ):
        """

        Parameters
        ----------
        plot : bool
                if true, plot the output.
        data_range : int
                Data range to use in the analysis.
        correlation_time : int
                Correlation time to use in the window sampling.
        integration_range : int
                Range over which integration should be performed.

        """
        self.plot = plot
        self.jacf: np.ndarray
        self.sigma = []

        if integration_range is None:
            integration_range = data_range - 1

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

        self.acfs = []
        self.sigmas = []

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
        # TODO improve docstring
        # Calculate the prefactor
        numerator = (elementary_charge**2) * (self.experiment.units.length**2)
        denominator = (
            3
            * boltzmann_constant
            * self.experiment.temperature
            * self.experiment.volume
            * self.experiment.units.volume
            * self.experiment.units.time
        )
        self.prefactor = numerator / denominator

    def ensemble_operation(self, ensemble: tf.Tensor):
        """
        Calculate and return the msd.

        Parameters
        ----------
        ensemble : tf.Tensor
                Ensemble on which to operate.

        Returns
        -------
        ACF of the tensor_values.

        """
        ensemble = tf.gather(ensemble, self.args.tau_values, axis=1)
        jacf = tfp.stats.auto_correlation(ensemble, normalize=False, axis=1, center=False)
        jacf = tf.squeeze(tf.reduce_sum(jacf, axis=-1), axis=0)
        self.sigmas.append(cumulative_trapezoid(jacf, x=self.time))

        return np.array(jacf)

    def _post_operation_processes(self):
        """
        Call the post-op processes
        Returns
        -------.

        """
        self.acf_array /= self.count
        sigma = cumulative_trapezoid(self.acf_array, x=self.time)
        sigma_SEM = np.std(self.sigmas, axis=0) / np.sqrt(len(self.sigmas))
        ionic_conductivity = self.prefactor * sigma[self.args.integration_range - 1]
        ionic_conductivity_SEM = (
            self.prefactor * sigma_SEM[self.args.integration_range - 1]
        )
        data = {
            self.result_keys[0]: [ionic_conductivity],
            self.result_keys[1]: [ionic_conductivity_SEM],
            self.result_series_keys[0]: self.time.tolist(),
            self.result_series_keys[1]: self.acf_array.tolist(),
            self.result_series_keys[2]: sigma.tolist(),
            self.result_series_keys[3]: sigma_SEM.tolist(),
        }

        self.queue_data(data=data, subjects=["System"])

    def plot_data(self, data):
        """Plot the data."""
        for selected_species, val in data.items():
            fig = figure(x_axis_label=self.x_label, y_axis_label=self.y_label)

            integral = np.array(val[self.result_series_keys[2]])
            integral_err = np.array(val[self.result_series_keys[3]])
            time = np.array(val[self.result_series_keys[0]])
            acf = np.array(val[self.result_series_keys[1]])
            # Compute the span
            span = Span(
                location=np.array(val[self.result_series_keys[0]])[
                    self.args.integration_range - 1
                ],
                dimension="height",
                line_dash="dashed",
            )
            # Compute vacf line
            fig.line(
                time,
                acf,
                color="#003f5c",
                legend_label=(
                    f"{selected_species}: {val[self.result_keys[0]][0]: 0.3E} +-"
                    f" {val[self.result_keys[1]][0]: 0.3E}"
                ),
            )

            fig.extra_y_ranges = {
                "Cond_Range": Range1d(start=0.6 * min(integral), end=1.3 * max(integral))
            }
            fig.add_layout(
                LinearAxis(
                    y_range_name="Cond_Range",
                    axis_label=r"$$\text{Ionic Conductivity} / Scm^{-1}$$",
                ),
                "right",
            )

            fig.line(time[1:], integral, y_range_name="Cond_Range", color="#bc5090")
            fig.varea(
                time[1:],
                integral - integral_err,
                integral + integral_err,
                alpha=0.3,
                color="#ffa600",
                y_range_name="Cond_Range",
            )

            fig.add_tools(HoverTool())
            fig.add_layout(span)
            self.plot_array.append(fig)

    def run_calculator(self):
        """Run analysis."""
        self.check_input()
        # Compute the pre-factor early.
        self._calculate_prefactor()

        dict_ref = str.encode(
            "/".join([DatasetKeys.OBSERVABLES, self.loaded_property.name])
        )
        self.count = 0
        self.acf_array = np.zeros((self.args.data_range,))
        batch_ds = self.get_batch_dataset([DatasetKeys.OBSERVABLES])
        for batch in tqdm(
            batch_ds,
            ncols=70,
            total=self.n_batches,
            disable=self.memory_manager.minibatch,
        ):
            ensemble_ds = self.get_ensemble_dataset(batch, DatasetKeys.OBSERVABLES)
            for ensemble in ensemble_ds:
                self.acf_array += self.ensemble_operation(ensemble[dict_ref])
                self.count += 1

        # Scale, save, and plot the data.
        self._post_operation_processes()
