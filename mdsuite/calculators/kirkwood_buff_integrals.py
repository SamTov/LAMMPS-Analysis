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
Module for the computation of kirkwood buff integrals.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from mdsuite.calculators.calculator import Calculator, call
from mdsuite.database.scheme import Computation
from mdsuite.utils.meta_functions import apply_savgol_filter

log = logging.getLogger(__name__)


@dataclass
class Args:
    """Data class for the saved properties."""

    savgol_order: int
    savgol_window_length: int
    number_of_bins: int
    number_of_configurations: int
    cutoff: float


class KirkwoodBuffIntegral(Calculator):
    """
    Class for the calculation of the Kirkwood-Buff integrals.

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
    data_files : list
                        list of files to be analyzed.
    rdf = None : list
                        rdf tensor_values being studied.
    radii = None : list
                        radii tensor_values corresponding to the rdf.
    species_tuple : list
                        A list of species combinations being studied.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run.KirkwoodBuffIntegral()

    """

    def __init__(self, **kwargs):
        """
        Python constructor for the class.

        Parameters
        ----------
        experiment : class object
                        Class object of the experiment.

        """
        super().__init__(**kwargs)
        self.file_to_study = None
        self.data_files = []
        self.rdf = None
        self.radii = None
        self.kb_integral = None
        self.database_group = "Kirkwood_Buff_Integral"
        self.x_label = r"$$ \text{r} / nm$$"
        self.y_label = r"$$\text{G}(\mathbf{r})$$"
        self.analysis_name = "Kirkwood-Buff_Integral"
        self.result_series_keys = ["r", "kb_integral"]
        self.data_range = 1

        self.post_generation = True

    @call
    def __call__(
        self,
        rdf_data: Computation = None,
        plot=True,
        savgol_order: int = 2,
        savgol_window_length: int = 17,
    ):
        """
        Call method for the KB integrals.

        Parameters
        ----------
        rdf_data : Computation
                MDSuite Computation data schema from which to load the RDF data and
                store relevant SQL meta-data information. If not give, an RDF will be
                computed using the default RDF arguments.
        plot : bool
                If true, the output will be displayed in a figure.
        savgol_order : int
                Order of the savgol polynomial filter
        savgol_window_length : int
                Window length of the savgol filter.

        """
        if isinstance(rdf_data, Computation):
            self.rdf_data = rdf_data
        else:
            self.rdf_data = self.experiment.run.RadialDistributionFunction(plot=False)
        self.plot = plot

        # set args that will affect the computation result
        self.args = Args(
            savgol_order=savgol_order,
            savgol_window_length=savgol_window_length,
            number_of_bins=self.rdf_data.computation_parameter["number_of_bins"],
            cutoff=self.rdf_data.computation_parameter["cutoff"],
            number_of_configurations=self.rdf_data.computation_parameter[
                "number_of_configurations"
            ],
        )

    def _calculate_kb_integral(self, radii_data: np.ndarray, rdf_data: np.ndarray):
        """
        Calculate the Kirkwood-Buff integral.

        Parameters
        ----------
        radii_data : np.ndarray
                Radii data to use in the computation.
        rdf_data : np.ndarray
                RDF data to use in the computation.

        Returns
        -------
        kb_integral : np.ndarray
                KB integral to be saved.

        """
        filtered_data = apply_savgol_filter(
            rdf_data,
            order=self.args.savgol_order,
            window_length=self.args.savgol_window_length,
        )
        integral_data = cumulative_trapezoid(
            y=(filtered_data[1:] - 1) * (radii_data[1:]) ** 2, x=radii_data[1:]
        )

        return 4 * np.pi * integral_data

    def run_calculator(self):
        """Calculate the potential of mean-force and perform error analysis."""
        for selected_species, vals in self.rdf_data.data_dict.items():
            selected_species = selected_species.split("_")

            radii = np.array(vals["x"]).astype(float)[1:]
            rdf = np.array(vals["y"]).astype(float)[1:]
            kb_integral = self._calculate_kb_integral(radii_data=radii, rdf_data=rdf)

            data = {
                self.result_series_keys[0]: radii[1:].tolist(),
                self.result_series_keys[1]: kb_integral.tolist(),
            }

            self.queue_data(data=data, subjects=selected_species)

    def plot_data(self, data):
        """Plot the data."""
        for selected_species, val in data.items():
            self.run_visualization(
                x_data=val[self.result_series_keys[0]],
                y_data=val[self.result_series_keys[1]],
                title=selected_species,
            )
