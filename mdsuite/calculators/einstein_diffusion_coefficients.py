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
Module for the computation of self-diffusion coefficients using the Einstein method.
"""
from __future__ import annotations

import logging
from abc import ABC
from typing import Any, List, Union

import numpy as np
import tensorflow as tf
from bokeh.models import HoverTool, LinearAxis, Span
from bokeh.models.ranges import Range1d
from bokeh.plotting import figure
from tqdm import tqdm

from mdsuite import utils
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.utils.calculator_helper_methods import fit_einstein_curve

log = logging.getLogger(__name__)


class EinsteinDiffusionCoefficients(TrajectoryCalculator, ABC):
    """
    Class for the Einstein diffusion coefficient implementation

    Attributes
    ----------
    msd_array : np.ndarray
            MSD data updated during each ensemble computation.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    project.experiment.run.EinsteinDiffusionCoefficients(data_range=500,
                                                         plot=True,
                                                         correlation_time=10)
    """

    def __init__(
        self,
        plot: bool = True,
        species: list = None,
        data_range: int = 100,
        correlation_time: int = 1,
        atom_selection: np.s_ = np.s_[:],
        molecules: bool = False,
        tau_values: Union[int, List, Any] = np.s_[:],
        fit_range: int = -1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        experiment :  Experiment
                Experiment class to call from
        experiments :  Experiment
                Experiment classes to call from
        """

        super().__init__(**kwargs)
        self.scale_function = {"linear": {"scale_factor": 150}}
        self.loaded_property = mdsuite_properties.unwrapped_positions
        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"$$\text{MSD} / m^{2}$$"
        self.result_keys = [
            "diffusion_coefficient",
            "uncertainty",
            "gradient",
            "intercept",
        ]
        self.result_series_keys = ["time", "msd", "gradients", "gradient_errors"]
        self.analysis_name = "Einstein Self-Diffusion Coefficients"
        self._dtype = tf.float64

        self.msd_array = None

        log.info("starting Einstein Diffusion Computation")

        if fit_range == -1:
            fit_range = int(data_range - 1)

        # set args that will affect the computation result
        self.stored_parameters = self.create_stored_parameters(
            data_range=data_range,
            correlation_time=correlation_time,
            atom_selection=atom_selection,
            tau_values=tau_values,
            molecules=molecules,
            species=species,
            fit_range=fit_range,
        )

        self.plot = plot
        self.system_property = False

    def prepare_calculation(self):
        """
        Helper method for parameters that need to be computed after the experiment
        attributes are exposed to the calculator.
        Returns
        -------

        """
        self.time = self._handle_tau_values()
        if self.stored_parameters.species is None:
            if self.stored_parameters.molecules:
                self.stored_parameters.species = list(self.experiment.molecules)
            else:
                self.stored_parameters.species = list(self.experiment.species)

    def ensemble_operation(self, ensemble):
        """
        Calculate and return the msd.

        Parameters
        ----------
        ensemble : tf.Tensor
                An ensemble of data to be operated on.

        Returns
        -------
        MSD of the tensor_values.
        """
        msd = tf.math.squared_difference(
            tf.gather(ensemble, self.stored_parameters.tau_values, axis=1),
            ensemble[:, None, 0],
        )
        # average over particles, sum over dimensions
        msd = tf.reduce_sum(tf.reduce_mean(msd, axis=0), axis=-1)
        # sum up ensembles to average in post processing
        self.msd_array += np.array(msd)

    def fit_diffusion_coefficients(self):
        """
        Apply unit conversion, fit line to the data, prepare for database storage
        """

        self.msd_array /= int(self.n_batches) * self.ensemble_loop
        self.msd_array *= self.experiment.units["length"] ** 2
        time = self.time * self.experiment.units["time"]
        self.msd_array *= self.experiment.units.length**2
        self.time *= self.experiment.units.time

        fit_values, covariance, gradients, gradient_errors = fit_einstein_curve(
            x_data=time,
            y_data=self.msd_array,
            fit_max_index=self.stored_parameters.fit_range,
        )
        error = np.sqrt(np.diag(covariance))[0]

        data = {
            self.result_keys[0]: 1 / 6.0 * fit_values[0],
            self.result_keys[1]: 1 / 6.0 * error,
            self.result_keys[2]: fit_values[0],
            self.result_keys[3]: fit_values[1],
            self.result_series_keys[0]: time.tolist(),
            self.result_series_keys[1]: self.msd_array.tolist(),
            self.result_series_keys[2]: (np.array(gradients) / 6).tolist(),
            self.result_series_keys[3]: (np.array(gradient_errors) / 6).tolist(),
        }
        return data

    def run_calculator(self):
        """
        Run analysis.
        """
        self._run_dependency_check()
        for species in self.stored_parameters.species:
            self.msd_array = np.zeros(self.data_resolution)
            dict_ref = str.encode("/".join([species, self.loaded_property.name]))
            batch_ds = self.get_batch_dataset([species])

            # loop over batches to get MSD
            for batch in tqdm(
                batch_ds,
                ncols=70,
                desc=species,
                total=self.n_batches,
                disable=self.memory_manager.minibatch,
            ):
                ensemble_ds = self.get_ensemble_dataset(batch, species)

                for ensemble in ensemble_ds:
                    self.ensemble_operation(ensemble[dict_ref])

            fit_results = self.fit_diffusion_coefficients()
            self.queue_data(data=fit_results, subjects=[species])

    def plot_data(self, data):
        """
        Plot the Einstein fits.

        Parameters
        ----------
        data

        Returns
        -------

        """
        plot_array = []
        for selected_species, val in data.data_dict.items():
            fig = figure(x_axis_label=self.x_label, y_axis_label=self.y_label)

            gradients = np.array(val[self.result_series_keys[2]])
            gradient_errors = np.array(val[self.result_series_keys[3]])

            time = np.array(val[self.result_series_keys[0]])
            msd = np.array(val[self.result_series_keys[1]])

            fig.y_range = Range1d(-0.0, 1.1 * max(msd))

            # Compute the span
            span = Span(
                location=time[self.stored_parameters.fit_range],
                dimension="height",
                line_dash="dashed",
            )
            # Compute msd and fit lines
            fig.line(
                time,
                msd,
                color=utils.Colour.ORANGE,
                legend_label=(
                    f"{selected_species}: {val[self.result_keys[0]]: 0.3E} +-"
                    f" {val[self.result_keys[1]]: 0.3E}"
                ),
            )
            fit_data = val[self.result_keys[2]] * time + val[self.result_keys[3]]
            fig.line(time, fit_data, color=utils.Colour.PAUA, legend_label="Curve fit.")
            fig.extra_y_ranges = {
                "Diff_range": Range1d(
                    start=0.999 * min(gradients), end=1.001 * max(gradients)
                )
            }

            fig.add_layout(
                LinearAxis(
                    y_range_name="Diff_range",
                    axis_label=r"$$\text{Diffusion Coefficient} / m^{2}s^{-1}$$",
                ),
                "right",
            )
            grad_time = time[-len(gradients) :]
            fig.line(
                grad_time,
                gradients,
                y_range_name="Diff_range",
                color=utils.Colour.MULBERRY,
            )
            fig.varea(
                grad_time,
                gradients - gradient_errors,
                gradients + gradient_errors,
                alpha=0.3,
                color=utils.Colour.ORANGE,
                y_range_name="Diff_range",
            )
            fig.add_tools(HoverTool())
            fig.add_layout(span)

            plot_array.append(fig)

        return plot_array
