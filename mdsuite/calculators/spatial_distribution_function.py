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
Module for the spatial distribution function calculator.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.calculators.calculator import call
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.utils.linalg import (
    cartesian_to_spherical_coordinates,
    get2dHistogram,
    spherical_to_cartesian_coordinates,
)
from mdsuite.utils.meta_functions import join_path
from mdsuite.utils.tensor_flow.layers import NLLayer
from mdsuite.visualizer.d3_data_visualizer import DataVisualizer3D

if TYPE_CHECKING:
    from mdsuite import Experiment

log = logging.getLogger(__name__)


@dataclass
class Args:
    """Data class for the saved properties."""

    number_of_configurations: int
    data_range: int
    correlation_time: int
    atom_selection: np.s_
    molecules: bool
    species: list
    r_min: float
    r_max: float
    n_bins: int


class SpatialDistributionFunction(TrajectoryCalculator):
    """Spatial Distribution Function Calculator based on the r_ij matrix."""

    def __init__(self, experiment: Experiment, experiments=None):
        """
        Constructor of the SpatialDistributionFunction.

        Parameters
        ----------
        experiment: Experiment
            managed by RunComputation
        experiments:
            list of Experiments, managed by RunComputation
        load_data: bool
            managed by RunComputation

        """
        super().__init__(experiment, experiments=experiments)

        self.scale_function = {"quadratic": {"outer_scale_factor": 1}}
        self.loaded_property = mdsuite_properties.positions
        self.x_label = r"$$\text{r} /  nm$$"  # None
        self.y_label = r"$$\text{g(r)}$$"  # None
        self.analysis_name = "Spatial_Distribution_Function"
        self.experimental = True

        self._dtype = tf.float32

    @call
    def __call__(
        self,
        molecules: bool = False,
        start: int = 1,
        stop: int = 10,
        number_of_configurations: int = 5,
        r_min: float = 4.0,
        r_max: float = 4.5,
        species: list = None,
        n_bins: int = 100,
        **kwargs,
    ):
        """
        User Interface to the Spatial Distribution Function.

        Parameters
        ----------
        molecules : bool
                If true, load molecules.
        start: int
            Index of the first configuration
        stop: int
            Index of the last configuration
        number_of_configurations: int
            Number of configurations to sample between start and stop
        r_min: float
            Minimal distance for the SDF
        r_max: float
            Maximal distance for the SDF
        species: list
            List of species to use, for computing the SDF,
            if None a single SDF of all available species will be computed
        kwargs

        """
        if species is None:
            if molecules:
                species = list(self.experiment.molecules)
            else:
                species = list(self.experiment.species)

        # choose sampled configurations
        self.sample_configurations = np.linspace(
            start, stop, number_of_configurations, dtype=np.int
        )
        self.plot = False

        self.args = Args(
            molecules=molecules,
            species=species,
            number_of_configurations=number_of_configurations,
            r_min=r_min,
            atom_selection=np.s_[:],
            r_max=r_max,
            data_range=number_of_configurations,
            correlation_time=1,
            n_bins=n_bins,
        )

    def _load_positions(self, indices: list, species: str) -> tf.Tensor:
        """
        Load the positions matrix.

        This function is here to optimize calculation speed

        Parameters
        ----------
        indices : list
                List of indices to take from the database_path
        species: str
                The species to load the positions from

        Returns
        -------
        loaded_data : tf.Tensor
                tf.Tensor of tensor_values loaded from the hdf5 database_path

        """
        path_list = [join_path(species, self.loaded_property.name)]

        data_dict = self.database.load_data(
            path_list=path_list, select_slice=np.s_[:, indices]
        )
        data = []
        for item in path_list:
            data.append(data_dict[item])
        if len(self.args.species) == 1:
            return tf.cast(data, dtype=self.dtype)
        else:
            return tf.cast(tf.concat(data, axis=0), dtype=self.dtype)

    def run_calculator(self):
        """Run the computation."""
        path_list = [
            join_path(item, self.loaded_property.name) for item in self.args.species
        ]
        self._prepare_managers(path_list)
        # Iterate over batches
        sdf_values = []

        nllayer = NLLayer()

        for idx, sample_configuration in tqdm(
            enumerate(np.array_split(self.sample_configurations, self.n_batches)),
            ncols=70,
        ):
            positions_tensor = []
            species_length = []
            for species in self.args.species:
                positions_tensor.append(
                    self._load_positions(sample_configuration, species)
                )
                species_length.append(len(positions_tensor[-1]))
                log.debug(f"Got {species_length[-1]} ions of {species}")

            positions_tensor = tf.concat(positions_tensor, axis=0)

            # make it (configurations, n_atoms, 3)
            positions_tensor = tf.transpose(positions_tensor, perm=(1, 0, 2))
            cell = tf.linalg.set_diag(tf.zeros((3, 3)), self.experiment.box_array)
            cell = tf.repeat(cell[None], positions_tensor.shape[0], axis=0)

            r_ij = nllayer({"positions": positions_tensor, "cell": cell})

            d_ij = tf.linalg.norm(r_ij, axis=-1)  # shape (b, i, j)
            # apply minimal and maximal distance and remove the diagonal elements of 0
            mask = (d_ij > self.args.r_min) & (d_ij < self.args.r_max)  # & (d_ij != 0)

            # Slicing the mask to the area where only the distances i!=j occur.
            # There are two such areas, so I am slicing them twice
            # could also mirror them
            mask_ = mask[:, species_length[0] :, : species_length[1]]
            r_ij_cut = r_ij[:, species_length[0] :, : species_length[1], :]
            r_ij_cut = r_ij_cut[mask_]
            sdf_values.append(self.r_ij_to_bins(r_ij_cut))
            # and the other half (only effective if species[0] != species[1])
            mask_ = mask[:, : species_length[0], species_length[1] :]
            r_ij_cut = r_ij[:, : species_length[0], species_length[1] :, :]
            r_ij_cut = r_ij_cut[mask_]
            sdf_values.append(self.r_ij_to_bins(r_ij_cut))

        sdf_values = tf.reduce_sum(sdf_values, axis=0)

        # TODO fix subjects and maybe rename
        self.queue_data(
            data={
                "sdf": sdf_values.numpy().tolist(),
                "sphere": self._get_unit_sphere().numpy().tolist(),
            },
            subjects=["System"],
        )

        coordinates = tf.reshape(self._get_unit_sphere(), [self.args.n_bins**2, 3])
        colour_map = tf.reshape(sdf_values, [-1])
        self._run_visualization(coordinates, colour_map)

    def _get_unit_sphere(self) -> tf.Tensor:
        """Get the coordinates on the sphere for the bins.

        Returns
        -------
        tf.Tensor:
            A Tensor with shape (n_bins, n_bins, 3) where 3 represents (x,y,z)
            for the coordinates of a unit sphere

        """
        theta_range = [0, math.pi]
        phi_range = [-math.pi, math.pi]
        theta_vals = np.linspace(theta_range[0], theta_range[1], self.args.n_bins)
        phi_vals = np.linspace(phi_range[0], phi_range[1], self.args.n_bins)

        xx, yy = np.meshgrid(theta_vals, phi_vals)
        spherical_map = tf.stack([tf.ones_like(xx), xx, yy], axis=-1)
        cartesian_map = spherical_to_cartesian_coordinates(spherical_map)

        return cartesian_map

    def r_ij_to_bins(self, r_ij) -> tf.Tensor:
        """Compute the 2D histogram in spherical coordinates while projecting
        all values of r to a unit sphere.

        Parameters
        ----------
        r_ij: tf.Tensor
            any  r_ij matrix with shape (..., 3)

        Returns
        -------
        tf.Tensor:
            bins with shape (n_bins, n_bins)

        """
        r_ij_spherical = cartesian_to_spherical_coordinates(r_ij)
        theta_phi_pairs = tf.reshape(r_ij_spherical, (-1, 3))

        theta_range = [0, math.pi]
        phi_range = [-math.pi, math.pi]

        bins = get2dHistogram(
            theta_phi_pairs[:, 1],
            theta_phi_pairs[:, 2],
            value_range=[theta_range, phi_range],
            nbins=self.args.n_bins,
        )

        return bins

    def _run_visualization(self, plot_data: tf.Tensor, colour_map: np.ndarray):
        """
        Run the visualizer.

        Parameters
        ----------
        plot_data : tf.Tensor
                Data to be plot.
        colour_map : tf.Tensor
                A colour map to highlight density on the unit sphere

        """
        if self.args.species[0] in list(self.experiment.species):
            center = self.args.species[0]
        else:
            center_dict = self.experiment.molecules[self.args.species[0]]["groups"]["0"]
            center = {}
            for item in center_dict:
                for index in center_dict[item]:
                    center[f"{item}_{index}"] = self.database.load_data(
                        path_list=[join_path(item, "Positions")],
                        select_slice=np.s_[index, 0],
                    )[join_path(item, "Positions")]
        visualizer = DataVisualizer3D(
            data=plot_data, title="SDF", center=center, colour_map=colour_map
        )
        visualizer.plot()
