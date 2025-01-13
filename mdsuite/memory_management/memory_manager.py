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
from typing import Tuple

import numpy as np
import tensorflow as tf

from mdsuite.database.simulation_database import Database
from mdsuite.utils import config
from mdsuite.utils.meta_functions import get_machine_properties, gpu_available
from mdsuite.utils.scale_functions import (
    linear_scale_function,
    linearithmic_scale_function,
    polynomial_scale_function,
    quadratic_scale_function,
)

log = logging.getLogger(__name__)


class MemoryManager:
    """
    A class to manage memory during MDSuite operations.

    In most MDSuite operations the memory requirements of a complete procedure will be
    greater than what is available to the average computer. Therefore, some degree of
    memory management is required for smooth running. This class should be instantiated
    during any operation so it can update the operation on what the current memory
    capabilities are.

    The class can work with several tensor_values-sets in the case an analysis requires
    this much tensor_values.

    Attributes
    ----------
    data_path : list
    database : Database
    parallel : bool
    memory_fraction : float
    scale_function : dict
    gpu : bool

    """

    def __init__(
        self,
        data_path: list = None,
        database: Database = None,
        parallel: bool = False,
        memory_fraction: float = 0.2,
        scale_function: dict = None,
        gpu: bool = gpu_available(),
        offset: int = 0,
    ):
        """
        Constructor for the memory manager.

        Parameters
        ----------
        data_path : list
                Path to the data to be used in a calculator. This is used when
                asking the database how big a dataset is.
                ["Na/Positions", "Ionic_Current/Ionic_Current"]
        database : Database
                Database class to be used for checking size parameters.
        parallel : bool
                If true, the memory should be computed for parallel processes.
        memory_fraction : float
                How much memory is the calculator allowed to use.
        scale_function : dict
                Scaling function to compute the memory scaling of a calculator.
        gpu : bool
                If true, gpu should be used.
        offset : int
                If data is being loaded from a non-zero point in the database the
                offset is used to take this into account. For example, expanding a
                transformation.

        """
        if scale_function is None:
            scale_function = {"linear": {"scale_factor": 10}}
        self.data_path = data_path
        self.parallel = parallel
        self.database = database
        self.memory_fraction = config.memory_fraction
        self.offset = offset

        self.machine_properties = get_machine_properties()
        if gpu:
            memory = 0
            for item in self.machine_properties["gpu"]:
                if self.machine_properties["gpu"][item]["memory"] > memory:
                    memory = self.machine_properties["gpu"][item]["memory"]

            self.machine_properties["memory"] = memory * 1e6
            tf.device("gpu")
        else:
            tf.device("cpu")

        self.batch_size = None
        self.n_batches = None
        self.remainder = None
        self.atom_batch_size = None
        self.n_atom_batches = None
        self.atom_remainder = None
        self.minibatch = False
        (
            self.scale_function,
            self.scale_function_parameters,
        ) = self._select_scale_function(scale_function)

    @staticmethod
    def _select_scale_function(input_dict: dict):
        """
        Select a scale function.

        Parameters
        ----------
        input_dict : dict
                Input dict which selects the correct function.

        Returns
        -------
        Updates the class state

        """

        def _string_to_function(argument: str):
            """
            Convert a string to a function choice.

            Parameters
            ----------
            argument : str
                    String to use in the matching.

            Returns
            -------
            function : Callable

            """
            switcher = {
                "linear": linear_scale_function,
                "log-linear": linearithmic_scale_function,
                "quadratic": quadratic_scale_function,
                "polynomial": polynomial_scale_function,
            }
            try:
                return switcher[argument]
            except KeyError:
                raise KeyError("Invalid choice")

        scale_function = _string_to_function(list(input_dict.keys())[0])
        scale_function_parameters = input_dict[list(input_dict.keys())[0]]

        return scale_function, scale_function_parameters

    def get_batch_size(self) -> tuple:
        """
        Calculate the batch size of an operation.

        This method takes the tensor_values requirements of an operation and returns
        how big each batch of tensor_values should be for such an operation.

        Returns
        -------
        batch_size : int
                number of elements that should be loaded at one time
        number_of_batches : int
                number of batches which can be loaded.
        remainder : int
                number of elements that will be left unloaded after a loop over all
                batches. This amount can then be loaded to collect unused tensor_values.

        """
        if self.data_path is None:
            raise ValueError("No tensor_values have been requested.")
        per_configuration_memory: float = 0.0
        for item in self.data_path:
            n_particles, n_configs, n_bytes = self.database.get_data_size(item)
            per_configuration_memory += n_bytes / n_configs
        per_configuration_memory = self.scale_function(
            per_configuration_memory, **self.scale_function_parameters
        )
        maximum_loaded_configurations = int(
            np.clip(
                (self.memory_fraction * self.machine_properties["memory"])
                / per_configuration_memory,
                1,
                n_configs - self.offset,
            )
        )
        batch_size = self._get_optimal_batch_size(maximum_loaded_configurations)
        number_of_batches, remainder = divmod((n_configs - self.offset), batch_size)
        self.batch_size = batch_size
        self.n_batches = number_of_batches
        self.remainder = remainder

        return batch_size, number_of_batches, remainder

    @staticmethod
    def hdf5_load_time(n: int):
        """
        Describes the load time of a hdf5 database i.e. O(log N).

        Parameters
        ----------
        n : int
                Amount of data to be loaded

        Returns
        -------
        load_time : float
                Load time of N data points from a hdf5 database.

        """
        return np.log(n)

    @staticmethod
    def _get_optimal_batch_size(naive_size):
        """
        Use the open/close and read speeds of the hdf5 database_path as well as the
        operation being performed to get an optimal batch size.

        Parameters
        ----------
        naive_size : int
                Naive batch size to be optimized

        Returns
        -------
        batch_size : int
                An optimized batch size

        """
        # db_io_time = self.database.get_load_time()
        return naive_size

    def _compute_atomwise_minibatch(self, data_range: int):
        """
        Compute the number of atoms which can be loaded in at one time.

        Parameters
        ----------
        data_range : int
                Data range used in the calculator.

        Returns
        -------
        Updates the following class attributes:
            self.batch_size : int
                    Size of each batch.
            self.n_batches : int
                    Number of batches to be looped over.
            self.remainder :
                    Remainder of batches after looping.
            self.n_atom_batches : int
                    Number of batches over atoms.
            self.atom_remainder : int
                    Remainder atoms after even batching.

        """
        per_atom_memory = 0  # memory usage per atom within ONE configuration
        per_configuration_memory = 0  # per configuration memory usage
        total_rows = 0
        for item in self.data_path:
            n_particles, n_configs, n_bytes = self.database.get_data_size(item)
            per_configuration_memory += n_bytes / n_configs
            per_atom_memory += per_configuration_memory / n_particles
            total_rows += n_particles

        # per_configuration_memory = self.scale_function(per_configuration_memory,
        # **self.scale_function_parameters)
        per_atom_memory = self.scale_function(
            per_atom_memory, **self.scale_function_parameters
        )
        fractions = [1 / 2, 1 / 4, 1 / 8, 1 / 20, 1 / 100, 1 / 200, 0]

        for (
            fraction
        ) in fractions:  # iterate over possible mini batch fractions to fit memory
            if fraction == 0:
                # All fractions failed, try atom wise mini batching - Set to one atom
                # at a time in the worst case.
                log.info(
                    "Could not find a good mini batch fraction - using single atom mini"
                    " batching!"
                )
                batch_size = int(
                    np.clip(
                        self.memory_fraction
                        * self.machine_properties["memory"]
                        / per_atom_memory,
                        1,
                        n_configs,
                    )
                )
                self.atom_batch_size = 1  # Set the mini batch size to single atom
                break

            atom_batch_memory = fraction * per_atom_memory
            batch_size = int(
                np.clip(
                    self.memory_fraction
                    * self.machine_properties["memory"]
                    / atom_batch_memory,
                    1,
                    n_configs,
                )
            )
            if (
                batch_size > data_range
            ):  # the batch size has to be larger than the data_range
                self.atom_batch_size = (
                    n_particles * fraction
                )  # Set the mini batch size to total_data_points * fraction
                break

        self.batch_size = batch_size
        self.n_batches = int(n_configs / batch_size)
        self.remainder = int(n_configs % batch_size)
        self.n_atom_batches = int(n_particles / self.atom_batch_size)
        self.atom_remainder = int(n_particles % self.atom_batch_size)

    def get_ensemble_loop(
        self, data_range: int, correlation_time: int = 1
    ) -> Tuple[int, bool]:
        """
        Get the tensor_values range partition quantity.

        Not only does tensor_values need to be batched, it then needs to be looped over
        in order to calculate some property. This method will return the number of
        loops possible given the tensor_values range and the correlation time

        Parameters
        ----------
        data_range : int
                Data range to be used in the analysis.
        correlation_time : int
                Correlation time to be considered when looping over the tensor_values

        Returns
        -------
        data_range_partitions : int
                Number of time the batch can be looped over with given data_range and
                correlation time.

        """
        final_window = self.batch_size - data_range
        if final_window < 0:
            self._compute_atomwise_minibatch(data_range)
            final_window = self.batch_size - data_range
            self.minibatch = True
            return int(np.clip(final_window / correlation_time, 1, None)), True

        else:
            return int(np.clip(final_window / correlation_time, 1, None)), False
