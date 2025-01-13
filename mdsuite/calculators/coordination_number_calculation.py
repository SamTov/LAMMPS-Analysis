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
Module to compute coodination numbers.
"""

import logging
from dataclasses import dataclass

import numpy as np
from bokeh.models import HoverTool, LinearAxis, Span
from bokeh.models.ranges import Range1d
from bokeh.plotting import figure
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks

from mdsuite import utils
from mdsuite.calculators.calculator import Calculator, call
from mdsuite.database.scheme import Computation
from mdsuite.utils.exceptions import CannotPerformThisAnalysis
from mdsuite.utils.meta_functions import apply_savgol_filter, golden_section_search

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


def _integrate_rdf(radii_data: np.array, rdf_data: np.array, density: float) -> np.array:
    """
    Integrate the rdf provided with appropriate pre-factors..

    Parameters
    ----------
    radii_data : np.ndarray
            A numpy array of radii data.
    rdf_data : np.array
            A numpy array of rdf data.
    density : float
            The density of the system for the species pair.

    Returns
    -------
    integral_data : np.array
            Cumulative integral of the RDF scaled by the radius and denisty.

    """
    integral_data = cumulative_trapezoid(
        y=radii_data[1:] ** 2 * rdf_data[1:], x=radii_data[1:]
    )

    return 4 * np.pi * density * integral_data


class CoordinationNumbers(Calculator):
    """
    Class for the calculation of coordination numbers.

    Attributes
    ----------
    experiment : class object
                Class object of the experiment.
    data_range : int (default=500)
                Range over which the property should be evaluated. This is not
                applicable to the current analysis as the full rdf will be
                calculated.
    x_label : str
                How to label the x axis of the saved plot.
    y_label : str
                How to label the y axis of the saved plot.
    analysis_name : str
                Name of the analysis. used in saving of the tensor_values and
                figure.
    file_to_study : str
                The tensor_values file corresponding to the rdf being studied.
    integral_data : list
                integrated rdf tensor_values.
    species_tuple : list
                A list of species combinations being studied.
    indices : list
                A list of indices which correspond to to correct coordination
                numbers.
    rdf_data : Computation
            RDF data from the user to use in the computation.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    .. code-block:: python

        experiment.run.CoordinationNumbers(
            savgol_order = 2, savgol_window_length = 17
        )

    """

    rdf_data: Computation

    def __init__(self, **kwargs):
        """
        Python constructor.

        Parameters
        ----------
        experiment : class object
                        Class object of the experiment.

        """
        super().__init__(**kwargs)
        self.file_to_study = None

        self.integral_data = None
        self.species_tuple = None
        self.indices = None

        self.post_generation = True

        self.database_group = "Coordination_Numbers"
        self.x_label = r"$$\text{r} /  nm$$"
        self.y_label = "CN"
        self.analysis_name = "Coordination_Numbers"
        self.result_keys = []
        self.result_series_keys = ["r", "cn"]

    @call
    def __call__(
        self,
        rdf_data: Computation = None,
        plot: bool = True,
        savgol_order: int = 2,
        savgol_window_length: int = 17,
        number_of_shells: int = 1,
    ):
        """

        Parameters
        ----------
        rdf_data : Computation (optional)
                MDSuite Computation data schema from which to load the RDF data and
                store relevant SQL meta-data information. If not give, an RDF will be
                computed using the default RDF arguments.
        plot : bool (default=True)
                            Decision to plot the analysis.
        savgol_order : int
                Order of the savgol polynomial filter
        savgol_window_length : int
                Window length of the savgol filter.
        number_of_shells : int
                Number of shells to look for.

        """
        if isinstance(rdf_data, Computation):
            self.rdf_data = rdf_data
        else:
            self.rdf_data = self.experiment.run.RadialDistributionFunction(plot=False)

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

        # Auto-populate the result keys.
        for i in range(self.args.number_of_shells):
            self.result_keys.append(f"CN_{i + 1}")
            self.result_keys.append(f"CN_{i + 1}_error")

        self._compute_nm_volume()

        self.plot = plot

    def _compute_nm_volume(self):
        """
        Compute the volume of the box in nm.

        Returns
        -------
        Updates the volume attribute of the class.

        """
        volume_si = self.experiment.volume * self.experiment.units.volume

        self.volume = volume_si / 1e-9**3

    def _get_density(self, species: str) -> float:
        """Use the species_tuple in front of the name for information about the pair."""
        species = species.split("_")  # get an array of the species being studied
        rdf_number_of_atoms = self.experiment.species[species[0]].n_particles

        return rdf_number_of_atoms / self.volume

    def _get_rdf_peaks(self, rdf: np.ndarray) -> np.ndarray:
        """
        Get the max values of the rdf for use in the minimum calculation.

        Parameters
        ----------
        rdf : np.ndarray
                A numpy array of rdf values.

        Returns
        -------
        peaks : np.ndarray
                If an exception is not raised, the function will return a list of peaks
                in the rdf.

        Raises
        ------
        ValueError
                Raised if the number of peaks required for the analysis are not met.

        """
        filtered_data = apply_savgol_filter(
            rdf,
            order=self.args.savgol_order,
            window_length=self.args.savgol_window_length,
        )
        peaks = find_peaks(filtered_data, height=1.0)[0]  # get the maximum values
        required_peaks = self.args.number_of_shells + 1

        # Check that the required number of peaks exist.
        if len(peaks) < required_peaks:
            msg = (
                "Not enough peaks were detecting in the RDF to perform the desired "
                "analysis. Try reducing the number of desired peaks or improving the "
                "quality of the RDF provided."
            )
            log.error(msg)
            raise CannotPerformThisAnalysis(msg)
        else:
            return peaks

    def _find_minima(self, radii: np.ndarray, rdf: np.ndarray) -> dict:
        """
        Use min finding algorithm to determine the minima of the function.

        Parameters
        ----------
        radii : np.ndarray
                A numpy array of radii values.
        rdf : np.ndarray
                A numpy array of rdf values.

        Returns
        -------
        coordination_shells : dict
                A dictionary of coordination shell radial ranges.

        """
        peaks = self._get_rdf_peaks(rdf)  # get the max value indices

        # Calculate the range in which the coordination numbers should exist.
        coordination_shells = {}
        for i in range(self.args.number_of_shells):
            coordination_shells[i] = np.zeros(2, dtype=int)
            cn_radii_range = golden_section_search(
                [radii, rdf], radii[peaks[i + 1]], radii[peaks[i]]
            )
            for j in range(2):
                coordination_shells[i][j] = np.where(radii == cn_radii_range[j])[0][0]

        return coordination_shells

    def _get_coordination_numbers(
        self, integral_data: np.ndarray, radii: np.ndarray, rdf: np.ndarray
    ) -> dict:
        """
        Calculate the coordination numbers.

        Parameters
        ----------
        integral_data : np.ndarray
                Integrated RDF data from which to compute the coordination numbers.
        radii : np.ndarray
                radii data to use in the analysis
        rdf : np.ndarray
                RDF data to use in the analysis

        Returns
        -------
        coordination_numbers : dict
                A dictionary of coordination numbers.

        """
        coordination_shells = self._find_minima(radii, rdf)  # get the minimums

        coordination_numbers = {}

        for key, val in coordination_shells.items():
            lower_bound = integral_data[val[0]]
            upper_bound = integral_data[val[1]]

            coordination_numbers[f"CN_{int(key) + 1}"] = np.mean(
                [lower_bound, upper_bound]
            )
            coordination_numbers[f"CN_{int(key) + 1}_error"] = np.std(
                [lower_bound, upper_bound]
            ) / np.sqrt(2)

        return coordination_numbers

    def run_calculator(self):
        """Calculate the coordination numbers and perform error analysis."""
        for selected_species, vals in self.rdf_data.data_dict.items():
            log.debug(f"Computing coordination numbers for {selected_species}")

            radii = np.array(vals["x"]).astype(float)[1:]
            rdf = np.array(vals["y"]).astype(float)[1:]

            selected_species = selected_species.split("_")

            density = self._get_density(selected_species[0])

            integral_data = _integrate_rdf(radii, rdf, density)

            coordination_numbers = self._get_coordination_numbers(
                integral_data, radii, rdf
            )

            data = {
                self.result_series_keys[0]: radii[1:].tolist(),
                self.result_series_keys[1]: integral_data.tolist(),
            }
            for item in self.result_keys:
                data[item] = coordination_numbers[item]

            self.queue_data(data=data, subjects=selected_species)

    def plot_data(self, data):
        """Plot the CN."""
        # Plot the values if required
        for selected_species, val in data.items():
            fig = figure(x_axis_label=self.x_label, y_axis_label=self.y_label)

            # Add vertical lines to the plot
            for i in range(self.args.number_of_shells):
                coordination_number = val[f"CN_{i + 1}"]
                index = np.argmin(
                    np.abs(
                        np.array(val[self.result_series_keys[1]]) - coordination_number
                    )
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

            # Add second axis and RDF plot
            rdf_radii = self.rdf_data[selected_species]["x"]
            rdf_gr = self.rdf_data[selected_species]["y"]
            fig.extra_y_ranges = {
                "g(r)": Range1d(start=0, end=int(max(rdf_gr[1:]) + 0.5))
            }
            fig.add_layout(
                LinearAxis(
                    y_range_name="g(r)",
                    axis_label="g(r)",
                ),
                "right",
            )

            fig.line(rdf_radii, rdf_gr, y_range_name="g(r)", color=utils.Colour.MULBERRY)

            self.plot_array.append(fig)
