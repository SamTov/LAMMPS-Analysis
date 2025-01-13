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
Parent class for the calculators.
"""

from __future__ import annotations

import functools
import logging
import warnings
from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import mdsuite.database.scheme as db
from mdsuite.database.calculator_database import CalculatorDatabase
from mdsuite.visualizer.d2_data_visualization import DataVisualizer2D

if TYPE_CHECKING:
    from mdsuite import Experiment

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


def call(func):
    """
    Decorator for the calculator call method.

    This decorator provides a unified approach for handling run_computation and
    load_data for a single or multiple experiments.
    It handles the `run.<calc>()` method, iterates over experiments and
    loads data if requested! Therefore, the __call__ method does not and can
    not return any values anymore!


    Notes
    -----
    When calling the calculator it will check if a computation with the given
    user arguments was already performed:
    >>> Calculator.get_computation_data() is not None

    if no computations are available it will
    1. prepare a database entry
    >>> Calculator.prepare_db_entry()
    2. save the user arguments
    >>> Calculator.save_computation_args()
    3. Run the analysis
    >>> Calculator.run_analysis()
    4. Save all the data to the database
    >>> Calculator.save_db_data()
    5. Finally query the the data from the database and pass them to the user / plotting
    >>> data = Calculator.get_computation_data()




    Parameters
    ----------
    func: Calculator.__call__ method

    Returns
    -------
    decorated __call__ method

    """

    @functools.wraps(func)
    def inner(self, *args, **kwargs) -> Union[db.Computation, Dict[str, db.Computation]]:
        """Manage the call method.

        Parameters
        ----------
        self: Calculator

        Returns
        -------
        data:
            A dictionary of shape {name: data} when called from the project class
            A list of [data] when called directly from the experiment class

        """
        # This is only true, when called via project.experiments.<exp>.run,
        #  otherwise the experiment will be None
        return_dict = self.experiment is None

        out = {}
        for experiment in self.experiments:
            CLS = self.__class__
            # NOTE: if the calculator accepts more than just experiment/experiments
            #  as init, this has to be changed!
            cls = CLS(experiment=experiment)
            # pass the user args to the calculator
            func(cls, *args, **kwargs)
            data = cls.get_computation_data()
            if data is None:
                # new calculation will be performed
                cls.prepare_db_entry()
                cls.save_computation_args()
                cls.run_analysis()
                cls.save_db_data()
                # Need to reset the user args, if they got change
                # or set to defaults, e.g. n_configurations = - 1 so
                # that they match the query
                func(cls, *args, **kwargs)
                data = cls.get_computation_data()

            if cls.plot:
                """Plot the data"""
                cls.plotter = DataVisualizer2D(
                    title=cls.analysis_name, path=experiment.figures_path
                )
                cls.plot_data(data.data_dict)
                cls.plotter.grid_show(cls.plot_array)

            out[cls.experiment.name] = data

        if return_dict:
            return out
        else:
            return out[self.experiment.name]

    return inner


class Calculator(CalculatorDatabase):
    """
    Parent class for analysis modules.

    Attributes
    ----------
    experiment : Experiment
                Experiment for which the calculator will be run.
    experiments : List[Experiment]
            List of experiments on which to run the calculator.
    plot : bool
            If true, the results will be plotted.
    system_property: bool (default = False)
            If the calculator returns a value for the whole system such as ionic
            conductivity or viscosity as opposed to a species-specific number.
    experimental : bool (default = False)
            If true, a warning is raised upon calling this calculator with more
            information about why it is experimental.
    selected_species: tuple
            Species currently being studied in a specific loop.
    analysis_name: str
            Name of the analysis to store in the database.
    time : np.ndarray
            Time array over which to integrate and plot.
    plotter : DataVisualizer2D
            Data visualizer class for use in the plotting.
    result_keys : list
            keys to use when storing the results. e.g.
            ["diffusion_coefficient", "uncertainty"]
    result_series_keys : list
            keys to use when storing series results e.g.
            ["time", "msd"]
    prefactor : float (optional)
            can be set if the same pre-factor is required many times.
    x_label : str
            x-label for the plots.
    y_label : str
            y-label for the plots.
    _dtype : object = tf.float64
            dtype required by the analysis.
    plot_array : list
            A list of plot objects to be show together at the end of the
            species loop.

    """

    def __init__(
        self, experiment: Experiment = None, experiments: List[Experiment] = None
    ):
        """
        Constructor for the calculator class.

        Parameters
        ----------
        experiment : Experiment
                Experiment for which the calculator will be run.
        experiments : List[Experiment]
                List of experiments on which to run the calculator.

        """
        # Set upon instantiation of parent class
        super().__init__(experiment)
        # NOTE: if the calculator accepts more than just experiment/experiments
        #  in the init the @call decorator has to be changed!
        self.experiment: Experiment = experiment
        self.experiments: List[Experiment] = experiments
        # Setting the experiment value supersedes setting experiments
        if self.experiment is not None:
            self.experiments = [self.experiment]

        self.plot = False

        # SQL data attributes.
        self.result_keys = None
        self.result_series_keys = None
        self.analysis_name = None
        self.selected_species = None

        # Calculator attributes
        self.system_property = False
        self.experimental = False
        self.time = None
        self.prefactor = None

        # Data attributes
        self._dtype: object = tf.float64

        # Plotting attributes
        self.plotter = None
        self.x_label = None
        self.y_label = None
        self.plot_array = []

    @property
    def dtype(self):
        """Get the dtype used for the calculator."""
        return self._dtype

    def run_visualization(
        self, x_data: np.ndarray, y_data: np.ndarray, title: str, layouts: object = None
    ):
        """
        Run a visualization session on the data.

        Parameters
        ----------
        layouts : object
                Additional plot features that may be added.
                See https://docs.bokeh.org/en/latest/docs/reference/models.html for
                more information.
        x_data : np.ndarray
                Data to be plotted along the x axis
        y_data : np.ndarray
                Data to be plotted along the y-axis
        title : str
                Title of the analysis.

        Returns
        -------
        Updates the plot array with a Bokeh plot object.

        """
        self.plot_array.append(
            self.plotter.construct_plot(
                x_data=x_data,
                y_data=y_data,
                title=title,
                x_label=self.x_label,
                y_label=self.y_label,
                layouts=layouts,
            )
        )

    def run_calculator(self):
        """
        Run the calculation. This should be implemented in each calculator.

        Returns
        -------

        """
        raise NotImplementedError

    def plot_data(self, data):
        """
        Plot the data coming from the database.

        Parameters
        ----------
        data: db.Compution.data_dict
                associated with the current project

        """
        for selected_species, val in data.items():
            self.run_visualization(
                x_data=np.array(val[self.result_series_keys[0]]),
                y_data=np.array(val[self.result_series_keys[1]]),
                title=(
                    f"{selected_species}: {val[self.result_keys[0]]: 0.3E} +-"
                    f" {val[self.result_keys[1]]: 0.3E}"
                ),
            )

    def run_analysis(self):
        """Run the appropriate analysis."""
        if self.experimental:
            log.warning(
                "This is an experimental calculator. Please see the "
                "documentation before using the results."
            )
        self.run_calculator()
