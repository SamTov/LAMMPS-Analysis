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
import logging
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.database import simulation_properties
from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path

log = logging.getLogger(__name__)


class ScaleCoordinates(Transformations):
    """
    Class to scale coordinates based on dumped index values

    Attributes
    ----------
    scale_function : dict
            A dictionary referencing the memory/time scaling function of the
            transformation.
    experiment : object
            Experiment this transformation is attached to.
    species : list
            Species on which this transformation should be applied.
    box : list
            Box vectors to multiply the indices by
    """

    def __init__(self, species: list = None):
        """Constructor for the Ionic current calculator.

        Parameters
        ----------
        species : list
                Species on which this transformation should be applied.
        """
        super().__init__()
        self.species = species
        self.scale_function = {"linear": {"scale_factor": 2}}
        self.dtype = tf.float64  # TODO should be a property that is immutable?

    def update_from_experiment(self):
        if self.species is None:
            self.species = list(self.experiment.species)

    def _check_for_indices(self):
        """
        Check the database_path for indices

        Returns
        -------

        """
        truth_table = []
        for item in self.species:
            path = join_path(item, "Scaled_Positions")
            truth_table.append(self.database.check_existence(path))

        if not all(truth_table):
            log.info(
                "Indices were not included in the database_path generation. Please"
                " check your simulation files."
            )
            sys.exit(1)

    def _transformation(self, data: tf.Tensor):
        """
        Apply the transformation to a batch of tensor_values.

        Parameters
        ----------
        data

        Returns
        -------
        Scaled coordinates : tf.Tensor
                Coordinates scaled by the image number.
        """
        return tf.math.multiply(data, self.experiment.box_array)

    def _prepare_database_entry(self, species: str):
        """
        Add the relevant datasets and groups in the database_path

        Parameters
        ----------
        species : str
                Species for which tensor_values will be added.
        Returns
        -------
        tensor_values structure for use in saving the tensor_values to the
        database_path.
        """
        path = join_path(species, "Positions")
        existing = self._run_dataset_check(path)
        if existing:
            old_shape = self.database.get_data_size(path)
            species_length = self.experiment.species[species].n_particles
            resize_structure = {
                path: (
                    species_length,
                    self.experiment.number_of_configurations - old_shape[0],
                    3,
                )
            }
            self.offset = old_shape[0]
            self.database.resize_dataset(
                resize_structure
            )  # add a new dataset to the database_path
            data_structure = {
                path: {
                    "indices": np.s_[:],
                    "columns": [0, 1, 2],
                    "length": species_length,
                }
            }
        else:
            species_length = self.experiment.species[species].n_particles
            number_of_configurations = self.experiment.number_of_configurations
            dataset_structure = {path: (species_length, number_of_configurations, 3)}
            self.database.add_dataset(dataset_structure)
            data_structure = {
                path: {
                    "indices": np.s_[:],
                    "columns": [0, 1, 2],
                    "length": species_length,
                }
            }

        return data_structure

    def _scale_coordinates(self):
        """
        Perform the unwrapping
        Returns
        -------
        Updates the database_path object.
        """
        for species in self.species:
            data_structure = self._prepare_database_entry(species)
            data_path = [join_path(species, "Scaled_Positions")]
            self._prepare_monitors(data_path)
            batch_generator, batch_generator_args = self.data_manager.batch_generator(
                dictionary=True
            )
            type_spec = {
                str.encode(data_path[0]): tf.TensorSpec(
                    shape=simulation_properties.scaled_positions[1], dtype=self.dtype
                ),
                str.encode("data_size"): tf.TensorSpec(shape=(), dtype=tf.int32),
            }
            data_set = tf.data.Dataset.from_generator(
                batch_generator, args=batch_generator_args, output_signature=type_spec
            )
            data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)
            for index, x in tqdm(
                enumerate(data_set), ncols=70, desc=f"{species}: Scaling Coordinates"
            ):
                data = self._transformation(x[str.encode(data_path[0])])
                self._save_coordinates(
                    data=data,
                    data_structure=data_structure,
                    index=index,
                    batch_size=self.batch_size,
                    system_tensor=False,
                    tensor=True,
                )

    def run_transformation(self):
        """
        Perform the transformation.
        """
        self._check_for_indices()
        self._scale_coordinates()  # run the transformation
