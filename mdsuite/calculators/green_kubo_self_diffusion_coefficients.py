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
Module for the computation of diffusion coefficients using the Green-Kubo approach.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from bokeh.models import HoverTool, LinearAxis, Span
from bokeh.models.ranges import Range1d
from bokeh.plotting import figure
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm

from mdsuite import utils
from mdsuite.calculators.calculator import call
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties


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


class GreenKuboDiffusionCoefficients(TrajectoryCalculator, ABC):
    """
    Class for the Green-Kubo diffusion coefficient implementation

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
            Property loaded from the database_path for the analysis.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.GreenKuboSelfDiffusionCoefficients(data_range=500,
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

        self.loaded_property = mdsuite_properties.velocities
        self.scale_function = {"linear": {"scale_factor": 150}}

        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"$$\text{VACF} / m^{2}s^{-2}$$"
        self.analysis_name = "Green Kubo Self-Diffusion Coefficients"
        self.result_keys = ["diffusion_coefficient", "uncertainty"]
        self.result_series_keys = ["time", "acf", "integral", "integral_uncertainty"]

        self._dtype = tf.float64

    @call
    def __call__(
        self,
        plot: bool = True,
        species: list = None,
        data_range: int = 500,
        correlation_time: int = 1,
        atom_selection=np.s_[:],
        molecules: bool = False,
        tau_values: Union[int, List, Any] = np.s_[:],
        integration_range: int = None,
    ):
        """
        Constructor for the Green-Kubo diffusion coefficients class.

        Parameters
        ----------
        plot : bool
                if true, plot the output.
        species : list
                List of species on which to operate.
        data_range : int
                Data range to use in the analysis.
        correlation_time : int
                Correlation time to use in the window sampling.
        atom_selection : np.s_
                Selection of atoms to use within the HDF5 database.
        molecules : bool
                If true, molecules are used instead of atoms.
        integration_range : int
                Range over which to integrate. Default is to integrate over
                the full data range.

        """
        if species is None:
            if molecules:
                species = list(self.experiment.molecules)
            else:
                species = list(self.experiment.species)
        if integration_range is None:
            integration_range = data_range - 1

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

        # Note: The following attributes are in SI units
        self.time = self._handle_tau_values() * self.experiment.units.time
        self.vacfs = []

    def check_input(self):
        """
        Check the user input to ensure no conflicts are present.

        Returns
        -------

        """
        self._run_dependency_check()

    def ensemble_operation(self, ensemble):
        """
        Calculate and return the msd.

        Parameters
        ----------
        ensemble

        Returns
        -------
        MSD of the tensor_values.

        """
        vacf = (
            self.experiment.units.length**2
            / self.experiment.units.time**2
            * tfp.stats.auto_correlation(ensemble, normalize=False, axis=1, center=False)
        )
        self.count += vacf.shape[0]

        # average particles, sum dimensions
        return_vacf = tf.reduce_sum(tf.reduce_sum(vacf, axis=0), -1)
        self.sigmas.append(
            cumulative_trapezoid(
                tf.reduce_sum(tf.reduce_mean(vacf, axis=0), -1), x=self.time
            )
        )

        return np.array(return_vacf)

    def plot_data(self, data: dict):
        """
        Plot the data.

        Parameters
        ----------
        data : dict
                Data loaded from the sql database to be plotted.

        """
        for selected_species, val in data.items():
            fig = figure(x_axis_label=self.x_label, y_axis_label=self.y_label)

            integral = np.array(val[self.result_series_keys[2]])
            integral_err = np.array(val[self.result_series_keys[3]])
            time = np.array(val[self.result_series_keys[0]])
            vacf = np.array(val[self.result_series_keys[1]])
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
                vacf,
                color=utils.Colour.PRUSSIAN_BLUE,
                legend_label=(
                    f"{selected_species}: {val[self.result_keys[0]][0]: 0.3E} +-"
                    f" {val[self.result_keys[1]][0]: 0.3E}"
                ),
            )

            fig.extra_y_ranges = {
                "Diff_range": Range1d(start=0.6 * min(integral), end=1.3 * max(integral))
            }
            fig.add_layout(
                LinearAxis(
                    y_range_name="Diff_range",
                    axis_label=r"$$\text{Diffusion Coefficient} / m^{2}s^{-1}$$",
                ),
                "right",
            )

            fig.line(
                time[1:], integral, y_range_name="Diff_range", color=utils.Colour.MULBERRY
            )
            fig.varea(
                time[1:],
                integral - integral_err,
                integral + integral_err,
                alpha=0.3,
                color=utils.Colour.ORANGE,
                y_range_name="Diff_range",
            )

            fig.add_tools(HoverTool())
            fig.add_layout(span)
            self.plot_array.append(fig)

    def postprocessing(self, species: str):
        """
        Apply post-processing to the data.

        Parameters
        ----------
        species : str
                Current species on which you are operating.

        Returns
        -------

        """
        self.acf_array = self.acf_array / self.count
        self.sigmas = np.array(self.sigmas)
        sigma = cumulative_trapezoid(self.acf_array, x=self.time)
        sigma_SEM = np.std(self.sigmas, axis=0) / np.sqrt(len(self.sigmas))

        diff_coeff = 1 / 3 * sigma[self.args.integration_range - 1]
        diff_coeff_SEM = 1 / 3 * sigma_SEM[self.args.integration_range - 1]

        data = {
            self.result_keys[0]: [diff_coeff],
            self.result_keys[1]: [diff_coeff_SEM],
            self.result_series_keys[0]: self.time.tolist(),
            self.result_series_keys[1]: self.acf_array.tolist(),
            self.result_series_keys[2]: sigma.tolist(),
            self.result_series_keys[3]: sigma_SEM.tolist(),
        }

        self.queue_data(data=data, subjects=[species])

    def run_calculator(self):
        """
        Run analysis.

        Returns
        -------

        """
        self.check_input()
        # Loop over species
        for species in self.args.species:
            dict_ref = str.encode("/".join([species, self.loaded_property.name]))

            batch_ds = self.get_batch_dataset([species])
            self.count = 0
            self.acf_array = np.zeros(self.data_resolution)
            self.sigmas = []

            for batch in tqdm(
                batch_ds,
                ncols=70,
                desc=species,
                total=self.n_batches,
                disable=self.memory_manager.minibatch,
            ):
                ensemble_ds = self.get_ensemble_dataset(batch, species)

                for ensemble in ensemble_ds:
                    if not ensemble[dict_ref].shape[1] == self.args.data_range:
                        continue
                    else:
                        self.acf_array += self.ensemble_operation(ensemble[dict_ref])
                        self.count += 1

            # Scale, save, and plot the data.
            self.postprocessing(species)
