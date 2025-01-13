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
A parent class for calculators that operate on the trajectory.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, List, Union

import numpy as np
import tensorflow as tf

import mdsuite.database.simulation_database
from mdsuite.calculators.transformations_reference import switcher_transformations
from mdsuite.database.data_manager import DataManager
from mdsuite.database.simulation_database import Database
from mdsuite.memory_management import MemoryManager
from mdsuite.utils.meta_functions import join_path

from .calculator import Calculator

if TYPE_CHECKING:
    from mdsuite import Experiment


class TrajectoryCalculator(Calculator, ABC):
    """
    Parent class for calculators operating on the trajectory.

    Attributes
    ----------
    data_resolution : int
            Resolution of the data to be plotted. This is necessary because if someone
            wants a data_range of 500 they may not want
    loaded_property : tuple
            The property being loaded from the simulation database.
    dependency : tuple
            A dependency required for the analysis to run.
    scale_function : dict
            The scaling behaviour of the computer. e.g.
            {"linear": {"scale_factor": 150}}.  See mdsuite.utils.scale_functions.py for
            the list of possible functions.
    batch_size : int
            Batch size to use. This is the number of configurations that can be loaded
            given the complexity and data requirements of the operation.
    n_batches : int
            Number of batches that can be looped over given the batch size.
    remainder : int
            The remainder of configurations after the batch process.
    minibatch : bool
            If true, atom-wise mini-batching will be used.
    memory_manager : MemoryManager
            Memory manager object to handle computation of batch sizes.
    data_manager : DataManager
            Data manager parent to handle preparation of data generators.
    _database : Database
            Simulation database from which data should be loaded.

    """

    def __init__(self, experiment: Experiment = None, experiments: List = None):
        """
        Constructor for the TrajectoryCalculator class.

        Parameters
        ----------
        experiment : Experiment
                Experiment for which the calculator will be run.
        experiments : List[Experiment]
                List of experiments on which to run the calculator.

        """
        super(TrajectoryCalculator, self).__init__(
            experiment=experiment, experiments=experiments
        )

        self.data_resolution = None
        self.loaded_property: mdsuite.database.simulation_database.PropertyInfo = None
        self.dependency: mdsuite.database.simulation_database.PropertyInfo = None
        self.scale_function = None
        self.batch_size: int = None
        self.n_batches: int = None
        self.remainder: int = None
        self.minibatch: bool = None
        self.memory_manager = None
        self.data_manager = None
        self._database = None

    @property
    def database(self):
        """Get the database based on the experiment database path."""
        if self._database is None:
            self._database = Database(self.experiment.database_path / "database.hdf5")
        return self._database

    def _run_dependency_check(self):
        """
        Check to see if the necessary property exists and build it if required.

        Returns
        -------
        Will call transformations if required.

        """
        if self.loaded_property is None:
            return

        if self.dependency is not None:
            dependency_exists = self.database.check_existence(self.dependency.name)
            if not dependency_exists:
                self._resolve_dependencies(self.dependency)

        loaded_property = self.database.check_existence(self.loaded_property.name)
        if not loaded_property:
            self._resolve_dependencies(self.loaded_property)

    def _resolve_dependencies(
        self, dependency: mdsuite.database.simulation_database.PropertyInfo
    ):
        """
        Resolve any calculation dependencies if possible.

        Parameters
        ----------
        dependency : str
                Name of the dependency to resolve.

        Returns
        -------

        """

        def _string_to_function(argument):
            """
            Select a transformation based on an input.

            Parameters
            ----------
            argument : str
                    Name of the transformation required

            Returns
            -------
            transformation call.

            """
            switcher_unwrapping = {"Unwrapped_Positions": self._unwrap_choice()}

            # add the other transformations and merge the dictionaries
            switcher = {**switcher_unwrapping, **switcher_transformations}

            try:
                return switcher[argument]
            except KeyError:
                raise KeyError("Data not in database and cannot be generated.")

        transformation = getattr(
            self.experiment.run, _string_to_function(dependency.name)
        )
        transformation()

    def _unwrap_choice(self):
        """
        Unwrap either with indices or with box arrays.

        Returns
        -------
        -------.

        """
        indices = self.database.check_existence("Box_Images")
        if indices:
            return "UnwrapViaIndices"
        else:
            return "CoordinateUnwrapper"

    def _handle_tau_values(self) -> np.array:
        """
        Handle the parsing of custom tau values.


        Returns
        -------
        times : np.array
            The time values corresponding to the selected tau values

        """
        if isinstance(self.args.tau_values, int):
            self.data_resolution = self.args.tau_values
            self.args.tau_values = np.linspace(
                0, self.args.data_range - 1, self.args.tau_values, dtype=int
            )
        if isinstance(self.args.tau_values, list) or isinstance(
            self.args.tau_values, np.ndarray
        ):
            self.data_resolution = len(self.args.tau_values)
            self.args.data_range = self.args.tau_values[-1] + 1
        if isinstance(self.args.tau_values, slice):
            self.args.tau_values = np.linspace(
                0, self.args.data_range - 1, self.args.data_range, dtype=int
            )[self.args.tau_values]
            self.data_resolution = len(self.args.tau_values)

        times = (
            np.asarray(self.args.tau_values)
            * self.experiment.time_step
            * self.experiment.sample_rate
        )

        return times

    def _check_remainder(self):
        """
        Check that the remainder is compatible with the calculator.

        It may come to pass that the remainder computed by the memory manager is not
        divisible by your data range. In this case, it must be clipped such that it is.

        Returns
        -------
        Updates the remainder attribute if required.

        """
        return self.remainder - (self.remainder % self.args.data_range)

    def _prepare_managers(self, data_path: list, correct: bool = False):
        """
        Prepare the memory and tensor_values monitors for calculation.

        Parameters
        ----------
        data_path : list
                List of tensor_values paths to load from the hdf5
                database_path.
        correct : bool


        Returns
        -------
        Updates the calculator class

        """
        self.memory_manager = MemoryManager(
            data_path=data_path,
            database=self.database,
            memory_fraction=0.8,
            scale_function=self.scale_function,
        )
        (
            self.batch_size,
            self.n_batches,
            self.remainder,
        ) = self.memory_manager.get_batch_size()
        self.ensemble_loop, self.minibatch = self.memory_manager.get_ensemble_loop(
            self.args.data_range, self.args.correlation_time
        )

        if self.minibatch:
            self.batch_size = self.memory_manager.batch_size
            self.n_batches = self.memory_manager.n_batches
            self.remainder = self.memory_manager.remainder

        self._check_remainder()

        if correct:
            self._correct_batch_properties()
        self.data_manager = DataManager(
            data_path=data_path,
            database=self.database,
            data_range=self.args.data_range,
            batch_size=self.batch_size,
            n_batches=self.n_batches,
            ensemble_loop=self.ensemble_loop,
            correlation_time=self.args.correlation_time,
            remainder=self.remainder,
            atom_selection=self.args.atom_selection,
            minibatch=self.minibatch,
            atom_batch_size=self.memory_manager.atom_batch_size,
            n_atom_batches=self.memory_manager.n_atom_batches,
            atom_remainder=self.memory_manager.atom_remainder,
        )

    def _correct_batch_properties(self):
        """
        Fix batch properties.

        Notes
        -----
        This method is called by some calculator

        """
        raise NotImplementedError

    def get_batch_dataset(
        self,
        subject_list: list = None,
        loop_array: np.ndarray = None,
        correct: bool = False,
    ) -> tf.data.Dataset:
        """
        Collect the batch loop dataset.

        Parameters
        ----------
        correct : bool
                If true, a calculator specific method is called to correct some
                of the batching properties. For example, the RDF code will over-ride
                the data range in favour of number of configurations as it does not
                require dynamic properties.
        subject_list : list (default = None)
                A str of subjects to collect data for in case this is necessary.
                e.g. subject = ['Na']
                     subject = ['Na', 'Cl', 'K']
                     subject = ['Ionic_Current']
        loop_array : np.ndarray (default = None)
                If this is not None, elements of this array will be looped over in
                in the batches which load data at their indices. For example,
                    loop_array = [[1, 4, 7], [10, 13, 16], [19, 21, 24]]
                In this case, in the fist batch, configurations 1, 4, and 7 will be
                loaded for the analysis. This is particularly important in the
                structural properties.

        Returns
        -------
        dataset : tf.data.Dataset
                A TensorFlow dataset for the batch loop to be iterated over.

        """
        path_list = [join_path(item, self.loaded_property.name) for item in subject_list]
        self._prepare_managers(path_list, correct=correct)
        type_spec = {}
        for item in subject_list:
            dict_ref = "/".join([item, self.loaded_property.name])
            type_spec[str.encode(dict_ref)] = tf.TensorSpec(
                shape=(None, None, self.loaded_property.n_dims), dtype=self.dtype
            )
        type_spec[str.encode("data_size")] = tf.TensorSpec(shape=(), dtype=tf.int32)

        batch_generator, batch_generator_args = self.data_manager.batch_generator(
            system=self.system_property, loop_array=loop_array
        )
        ds = tf.data.Dataset.from_generator(
            generator=batch_generator,
            args=batch_generator_args,
            output_signature=type_spec,
        )

        return ds.prefetch(tf.data.AUTOTUNE)

    def get_ensemble_dataset(self, batch: dict, subject: Union[str, list]):
        """
        Collect the ensemble loop dataset.

        Parameters
        ----------
        subject : str
                What object to loop over.
        batch : tf.Tensor
                A batch of data to be looped over in ensembles.

        Returns
        -------
        dataset : tf.data.Dataset
                A TensorFlow dataset object for the ensemble loop to be iterated over.

        """
        (
            ensemble_generator,
            ensemble_generators_args,
        ) = self.data_manager.ensemble_generator(
            glob_data=batch, system=self.system_property
        )

        type_spec = {}
        if isinstance(subject, str):
            loop_list = [subject]
        else:
            loop_list = subject
        for item in loop_list:
            dict_ref = "/".join([item, self.loaded_property.name])
            type_spec[str.encode(dict_ref)] = tf.TensorSpec(
                shape=(None, None, self.loaded_property.n_dims), dtype=self.dtype
            )

        ds = tf.data.Dataset.from_generator(
            generator=ensemble_generator,
            args=ensemble_generators_args,
            output_signature=type_spec,
        )

        return ds.prefetch(tf.data.AUTOTUNE)
