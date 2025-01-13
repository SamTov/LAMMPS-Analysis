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
Module for the data manager. The data manager handles loading of data as TensorFlow
generators. These generators allow for the full use of the TF data pipelines but can
required special formatting rules.
"""

import logging

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.database.simulation_database import Database

log = logging.getLogger(__name__)


class DataManager:
    """
    Class for the MDS tensor_values fetcher.

    Due to the amount of tensor_values that needs to be collected and the possibility
    to optimize repeated loading, a separate tensor_values fetching class is required.
    This class manages how tensor_values is loaded from the MDS database_path and
    optimizes processes such as pre-loading and parallel reading.
    """

    def __init__(
        self,
        database: Database = None,
        data_path: list = None,
        data_range: int = None,
        n_batches: int = None,
        batch_size: int = None,
        ensemble_loop: int = None,
        correlation_time: int = 1,
        remainder: int = None,
        atom_selection=np.s_[:],
        minibatch: bool = False,
        atom_batch_size: int = None,
        n_atom_batches: int = None,
        atom_remainder: int = None,
        offset: int = 0,
    ):
        """
        Constructor for the DataManager class.

        Parameters
        ----------
        database : Database
                Database object from which tensor_values should be loaded
        data_path : list
                Path in the HDF5 database to be loaded.
        data_range : int
                Data range used in the calculator.
        n_batches : int
                Number of batches required.
        batch_size : int
                Size of a batch.
        ensemble_loop : int
                Number of ensembles to be looped over.
        correlation_time : int
                Correlation time used in the calculator.
        remainder : int
                Remainder used in the batching.
        atom_remainder : int
                Atom-wise remainder used in the atom-wise batching.
        minibatch : bool
                If true, atom-wise batching is required.
        atom_batch_size : int
                Size of an atom-wise batch.
        n_atom_batches : int
                Number of atom-wise batches.
        atom_selection : int
                Selection of atoms in the calculation.
        offset : int
                Offset in the data loading if it should not be loaded from the start.

        """
        self.database = database
        self.data_path = data_path
        self.minibatch = minibatch
        self.atom_batch_size = atom_batch_size
        self.n_atom_batches = n_atom_batches
        self.atom_remainder = atom_remainder
        self.offset = offset

        self.data_range = data_range
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.remainder = remainder
        self.ensemble_loop = ensemble_loop
        self.correlation_time = correlation_time
        self.atom_selection = atom_selection

    def batch_generator(  # noqa: C901
        self,
        dictionary: bool = False,
        system: bool = False,
        remainder: bool = False,
        loop_array: np.ndarray = None,
    ) -> tuple:
        """
        Build a generator object for the batch loop.

        Parameters
        ----------
        dictionary : bool
                If true return a dict. This is default now and could be removed.
        system : bool
                If true, a system parameter is being called for.
        remainder : bool
                If true, a remainder batch must be computed.
        loop_array : np.ndarray
                If this is not None, elements of this array will be looped over in
                in the batches which load data at their indices. For example,
                    loop_array = [[1, 4, 7], [10, 13, 16], [19, 21, 24]]
                In this case, in the fist batch, configurations 1, 4, and 7 will be
                loaded for the analysis. This is particularly important in the
                structural properties.

        Returns
        -------
        Returns a generator function and its arguments

        """
        args = (
            self.n_batches,
            self.batch_size,
            self.database.path,
            self.data_path,
            dictionary,
        )

        def generator(
            batch_number: int,
            batch_size: int,
            database: str,
            data_path: list,
            dictionary: bool,
        ):
            """
            Generator function for the batch loop.

            Parameters
            ----------
            batch_number : int
                    Number of batches to be looped over
            batch_size : int
                    size of each batch to load
            database : Database
                    database_path from which to load the tensor_values
            data_path : str
                    Path to the tensor_values in the database_path
            dictionary : bool
                    If true, tensor_values is returned in a dictionary

            Returns
            -------

            """
            database = Database(database)

            loop_over_remainder = self.remainder > 0

            for batch in range(batch_number + int(loop_over_remainder)):
                start = int(batch * batch_size) + self.offset
                stop = int(start + batch_size)
                data_size = tf.cast(batch_size, dtype=tf.int32)
                # Handle the remainder
                if batch == batch_number:
                    stop = int(start + self.remainder)
                    data_size = tf.cast(self.remainder, dtype=tf.int16)
                    # TODO make default

                if loop_array is not None:
                    if isinstance(self.atom_selection, dict):
                        select_slice = {}
                        for item in self.atom_selection:
                            select_slice[item] = np.s_[
                                self.atom_selection[item], loop_array[batch]
                            ]
                    else:
                        select_slice = np.s_[self.atom_selection, loop_array[batch]]
                elif system:
                    select_slice = np.s_[start:stop]
                else:
                    if type(self.atom_selection) is dict:
                        select_slice = {}
                        for item in self.atom_selection:
                            select_slice[item] = np.s_[
                                self.atom_selection[item], start:stop
                            ]
                    else:
                        select_slice = np.s_[self.atom_selection, start:stop]

                yield database.load_data(
                    data_path,
                    select_slice=select_slice,
                    dictionary=dictionary,
                    d_size=data_size,
                )

        def atom_generator(
            batch_number: int,
            batch_size: int,
            database: str,
            data_path: list,
            dictionary: bool,
        ):
            """
            Generator function for a mini-batched calculation.

            Parameters
            ----------
            batch_number : int
                    Number of batches to be looped over
            batch_size : int
                    size of each batch to load
            database : Database
                    database_path from which to load the tensor_values
            data_path : str
                    Path to the tensor_values in the database_path
            dictionary : bool
                    If true, tensor_values is returned in a dictionary

            Returns
            -------

            """
            # Atom selection not currently available for mini-batched calculations
            if type(self.atom_selection) is dict:
                raise ValueError(
                    "Atom selection is not currently available "
                    "for mini-batched calculations"
                )

            database = Database(database)
            _atom_remainder = [1 if self.atom_remainder else 0][0]
            start = 0
            for atom_batch in tqdm(
                range(self.n_atom_batches + _atom_remainder),
                total=self.n_atom_batches + _atom_remainder,
                ncols=70,
                desc="batch loop",
            ):
                atom_start = atom_batch * self.atom_batch_size
                atom_stop = atom_start + self.atom_batch_size
                if atom_batch == self.n_atom_batches:
                    atom_stop = start + self.atom_remainder
                for batch in range(batch_number + int(remainder)):
                    start = int(batch * batch_size) + self.offset
                    stop = int(start + batch_size)
                    data_size = tf.cast(batch_size, dtype=tf.int32)
                    if batch == batch_number:
                        stop = int(start + self.remainder)
                        data_size = tf.cast(self.remainder, dtype=tf.int16)
                    select_slice = np.s_[int(atom_start) : int(atom_stop), start:stop]
                    yield database.load_data(
                        data_path,
                        select_slice=select_slice,
                        dictionary=dictionary,
                        d_size=data_size,
                    )

        if self.minibatch:
            return atom_generator, args
        else:
            return generator, args

    def ensemble_generator(self, system: bool = False, glob_data: dict = None) -> tuple:
        """
        Build a generator for the ensemble loop.

        Parameters
        ----------
        system : bool
                If true, the system generator is returned.
        glob_data : dict
                data to be loaded in ensembles from a tensorflow generator.
                e.g. {b'Na/Positions': tf.Tensor}.
                Will usually include a b'data_size' key which is checked in the
                loop and ignored. All keys are in byte arrays. This appears when you
                pass a dict to the tensorflow generator.

        Returns
        -------
        Ensemble loop generator

        """
        args = (self.ensemble_loop, self.correlation_time, self.data_range)

        def dictionary_generator(ensemble_loop, correlation_time, data_range):
            """
            Generator for the ensemble loop

            Parameters
            ----------
            ensemble_loop : int
                    Number of ensembles to loop over
            correlation_time : int
                    Distance between ensembles
            data_range : int
                    Size of each ensemble

            Returns
            -------
            None.

            """
            ensemble_loop = int(
                np.clip(
                    (glob_data[b"data_size"] - data_range) / correlation_time, 1, None
                )
            )
            for ensemble in range(ensemble_loop):
                start = ensemble * correlation_time
                stop = start + data_range
                output_dict = {}
                for item in glob_data:
                    if item == str.encode("data_size"):
                        pass
                    else:
                        output_dict[item] = glob_data[item][:, start:stop]

                yield output_dict

        return dictionary_generator, args
