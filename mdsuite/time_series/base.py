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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from mdsuite.database.simulation_database import Database

if TYPE_CHECKING:
    from mdsuite import Experiment


def running_mean(x, size):
    """Perform a rolling window mean."""
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[size:] - cumsum[:-size]) / float(size)


class TimeSeries:
    """Time Series class."""

    def __init__(self, experiment: Experiment):
        """

        Parameters
        ----------
        experiment: Experiment
            The parent experiment class to perform the time series operation on

        """
        self.experiment = experiment

        self.loaded_property = None
        self.fig_labels = {"x": None, "y": None}
        self.species = experiment.species
        self.rolling_window = 0
        self.reduce_sum = True

        # Properties
        self._database = None
        self._data = None

    def __call__(self, species: list = None, rolling_window: int = 0):
        if species is not None:
            self.species = species
        self.rolling_window = rolling_window
        self.plot()

    @property
    def database(self):
        """Get the database."""
        if self._database is None:
            self._database = Database(self.experiment.database_path / "database.hdf5")
        return self._database

    @property
    def data(self):
        """Get the data for all species and timesteps for the loaded_property."""
        if self._data is None:
            self._data = tf.concat(
                [
                    self.database.load_data([f"{species}/{self.loaded_property}"])
                    for species in self.species
                ],
                axis=0,
            )
        return self._data

    @property
    def preprocess_data(self):
        """Perform some data preprocessing before plotting it."""
        data = self.data
        if self.reduce_sum:
            data = tf.einsum("atx -> t", data)
            # perform a reduce sum over atoms "a" and property dimension "x" to
            # yield time steps "t"
        if self.rolling_window > 0:
            data = running_mean(data, self.rolling_window)

        return data

    def plot(self):
        """Plot the data over timesteps."""
        fig, ax = plt.subplots()
        ax.plot(self.preprocess_data)
        ax.set_xlabel(self.fig_labels["x"])
        ax.set_ylabel(self.fig_labels["y"])
        fig.show()
