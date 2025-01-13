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
Module for the computation of the potential of mean force (PMF). The PMF can be used to
better understand effective bond strength between species of a system.
"""

import logging
from dataclasses import dataclass

import numpy as np
from bokeh.models import HoverTool, Span
from bokeh.plotting import figure
from scipy.signal import find_peaks

from mdsuite import utils
from mdsuite.calculators.calculator import Calculator, call
from mdsuite.database.scheme import Computation
from mdsuite.utils.meta_functions import apply_savgol_filter, golden_section_search
from mdsuite.utils.units import boltzmann_constant

log = logging.getLogger(__name__)


@dataclass
class Args:
    """Data class for the saved properties."""

    savgol_order: int
    savgol_window_length: int
    number_of_bins: int
    number_of_configurations: int
    cutoff: float
    number_of_shells: int


class PotentialOfMeanForce(Calculator):
    """
    Class for the calculation of the potential of mean-force.

    The potential of mean-force is a measure of the binding strength between
    atomic species in a experiment.

    Attributes
    ----------
    experiment : class object
                        Class object of the experiment.
    data_range : int (default=500)
                        Range over which the property should be evaluated.
                        This is not applicable to the current analysis as the
                        full rdf will be calculated.
    x_label : str
                        How to label the x axis of the saved plot.
    y_label : str
                        How to label the y axis of the saved plot.
    analysis_name : str
                        Name of the analysis. used in saving of the
                        tensor_values and figure.
    file_to_study : str
                        The tensor_values file corresponding to the rdf being
                        studied.
    data_files : list
                        list of files to be analyzed.
    rdf = None : list
                        rdf tensor_values being studied.
    radii = None : list
                        radii tensor_values corresponding to the rdf.
    selected_species : list
                        A list of species combinations being studied.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.PotentialOfMeanForce(savgol_order = 2,
                                                    savgol_window_length = 17)

    """

    def __init__(self, **kwargs):
        """
        Python constructor for the class.

        Parameters
        ----------
        experiment : class object
                        Class object of the experiment.
        experiments : class object
                        Class object of the experiment.
        load_data : bool

        """
        super().__init__(**kwargs)
        self.file_to_study = None
        self.rdf = None
        self.radii = None
        self.pomf = None
        self.indices = None
        self.x_label = r"$$r /  nm$$"
        self.y_label = r"$$w^{2}(r)$$"
        self.data_range = 1

        self.result_keys = []
        self.result_series_keys = ["r", "pomf"]

        self.analysis_name = "Potential_of_Mean_Force"
        self.post_generation = True

    @call
    def __call__(
        self,
        rdf_data: Computation = None,
        plot=True,
        savgol_order: int = 2,
        savgol_window_length: int = 17,
        number_of_shells: int = 1,
    ):
        """
        Python constructor for the class.

        Parameters
        ----------
        rdf_data : Computation
                RDF data to use in the computation.
        plot : bool (default=True)
                            Decision to plot the analysis.
        savgol_order : int
                Order of the savgol polynomial filter
        savgol_window_length : int
                Window length of the savgol filter.
        number_of_shells : int
                Number of shells to integrate through.

        """
        if isinstance(rdf_data, Computation):
            self.rdf_data = rdf_data
        else:
            self.rdf_data = self.experiment.run.RadialDistributionFunction(plot=False)

        self.plot = plot
        self.data_files = []

        # set args that will affect the computation result
        self.args = Args(
            savgol_order=savgol_order,
            savgol_window_length=savgol_window_length,
            number_of_bins=self.rdf_data.computation_parameter["number_of_bins"],
            cutoff=self.rdf_data.computation_parameter["cutoff"],
            number_of_configurations=self.rdf_data.computation_parameter[
                "number_of_configurations"
            ],
            number_of_shells=number_of_shells,
        )

        # Auto-populate the results.
        for i in range(self.args.number_of_shells):
            self.result_keys.append(f"POMF_{i + 1}")
            self.result_keys.append(f"POMF_{i + 1}_error")

    def _calculate_potential_of_mean_force(self, rdf: np.ndarray) -> np.ndarray:
        """
        Calculate the potential of mean force.

        Parameters
        ----------
        rdf : np.ndarray
                RDF data to use in the computation.

        Returns
        -------
        pomf : np.ndarray
                The computed pomf array.

        Notes
        -----
        Units here are always eV as the data stored in the RDF is constant independent
        of what was in the simulation.

        """
        pomf = -1 * boltzmann_constant * self.experiment.temperature * np.log(rdf)

        return pomf * 6.242e8  # convert to eV

    def _populate_args(self) -> tuple:
        """
        Use the provided RDF data to populate the args class.

        Returns
        -------
        number_of_bins : int
                The data range used in the RDF calculation.
        cutoff : float
                The cutoff (in nm) used in the RDF calculation

        """
        raw_data = self.rdf_data.data_dict
        keys = list(raw_data)
        number_of_bins = len(raw_data[keys[0]]["x"])
        cutoff = raw_data[keys[0]]["x"][-1]

        return number_of_bins, cutoff

    def get_pomf_peaks(self, pomf_data: np.ndarray) -> np.ndarray:
        """
        Calculate the maximums of the rdf.

        Parameters
        ----------
        pomf_data : np.ndarray
                POMF data to use in the peak detection.

        Returns
        -------
        peaks : np.ndarray
                Peaks to be used in the calculation.

        Raises
        ------
        ValueError
                Raised if the number of peaks required for the analysis are not met.

        """
        filtered_data = apply_savgol_filter(
            pomf_data,
            order=self.args.savgol_order,
            window_length=self.args.savgol_window_length,
        )

        required_peaks = self.args.number_of_shells + 1

        # Find the maximums in the filtered dataset
        peaks = find_peaks(filtered_data)[0]

        # Check that the required number of peaks exist.
        if len(peaks) < required_peaks:
            msg = (
                "Not enough peaks were detecting in the RDF to perform the desired "
                "analysis. Try reducing the number of desired peaks or improving the "
                "quality of the RDF provided."
            )
            log.error(msg)
            raise ValueError(msg)
        else:
            return peaks

    def _find_minimum(self, pomf_data: np.ndarray, radii_data: np.ndarray) -> dict:
        """
        Find the minimum of the pomf function.

        This function calls an implementation of the Golden-section search
        algorithm to determine the minimum of the potential of mean-force function.

        Parameters
        ----------
        pomf_data : np.ndarray
                POMF data to use in the min finding.
        radii_data : np.ndarray
                Radii data to use in the min finding.

        Returns
        -------
        pomf_shells : dict
                Dict of all shells detected based on user arguments, e.g:
                {'1': [0.1, 0.2]} indicates that the first pomf peak is betwee 0.1 and
                0.2 angstrom.

        """
        # get the peaks of the tensor_values post-filtering
        peaks = self.get_pomf_peaks(pomf_data)

        pomf_shells = {}
        for i in range(self.args.number_of_shells):
            pomf_shells[i] = np.zeros(2, dtype=int)
            pomf_radii_range = golden_section_search(
                [radii_data, pomf_data], radii_data[peaks[i + 1]], radii_data[peaks[i]]
            )
            for j in range(2):
                pomf_shells[i][j] = np.where(radii_data == pomf_radii_range[j])[0][0]

        return pomf_shells

    def _get_pomf_values(self, pomf: np.ndarray, radii: np.ndarray) -> dict:
        """
        Use a min-finding algorithm to calculate pomf values along the curve.

        Parameters
        ----------
        pomf : np.ndarray
                POMF function from which to compute properties.
        radii : np.ndarray
                Array of radii values to use in the min finding.

        Returns
        -------
        pomf_data : dict
                A dictionary of the pomf values and their uncertainty. e,g:
                {"POMF_1": 5.6, "POMF_1_error": 0.01}

        """
        pomf_shells = self._find_minimum(pomf, radii)

        pomf_data = {}
        for key, val in pomf_shells.items():
            lower_bound = pomf[val[0]]
            upper_bound = pomf[val[1]]
            pomf_data[f"POMF_{int(key) + 1}"] = np.mean([lower_bound, upper_bound])
            pomf_data[f"POMF_{int(key) + 1}_error"] = np.std(
                [lower_bound, upper_bound]
            ) / np.sqrt(2)

        return pomf_data

    def run_calculator(self):
        """Calculate the potential of mean-force and perform error analysis."""
        for selected_species, vals in self.rdf_data.data_dict.items():
            selected_species = selected_species.split("_")
            radii = np.array(vals["x"]).astype(float)[1:]
            rdf = np.array(vals["y"]).astype(float)[1:]

            log.debug(f"rdf: {rdf} \t radii: {radii}")
            pomf = self._calculate_potential_of_mean_force(rdf)

            pomf_data = self._get_pomf_values(pomf, radii)

            data = {
                self.result_series_keys[0]: radii[1:].tolist(),
                self.result_series_keys[1]: pomf.tolist(),
            }
            for item in self.result_keys:
                data[item] = pomf_data[item]

            self.queue_data(data=data, subjects=selected_species)

    def plot_data(self, data):
        """Plot the POMF."""
        log.debug("Start plotting the POMF.")
        for selected_species, val in data.items():
            fig = figure(x_axis_label=self.x_label, y_axis_label=self.y_label)

            # Add vertical lines to the plot
            for i in range(self.args.number_of_shells):
                pomf_value = val[f"POMF_{i + 1}"]
                index = np.argmin(
                    np.abs(np.array(val[self.result_series_keys[1]]) - pomf_value)
                )
                r_location = val[self.result_series_keys[0]][index]
                span = Span(location=r_location, dimension="height", line_dash="dashed")
                fig.add_layout(span)

            # Add the CN line and hover tool
            fig.line(
                val[self.result_series_keys[0]],
                val[self.result_series_keys[1]],
                color=utils.Colour.PRUSSIAN_BLUE,
                # legend labels are always the first shell and first shell error.
                legend_label=(
                    f"{selected_species}: {val[self.result_keys[0]]: 0.3E} +-"
                    f" {val[self.result_keys[1]]: 0.3E}"
                ),
            )
            fig.add_tools(HoverTool())

            self.plot_array.append(fig)
