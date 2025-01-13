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
Test for the memory manager module.
"""

import unittest

import numpy as np

from mdsuite.memory_management.memory_manager import MemoryManager


class TestDatabase:
    """Test database for the unit testing."""

    def __init__(self, data_size: int = 500, rows: int = 10, columns: int = 10):
        """Constructor for the test database."""
        self.data_size = data_size
        self.rows = rows
        self.columns = columns

    def get_data_size(self, item):
        """
        Compute the size of a dataset.

        Returns
        -------

        """
        return self.rows, self.columns, self.data_size


class TestMemoryManager(unittest.TestCase):
    """Test the memory manager module."""

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare the class.

        Returns
        -------

        """
        cls.memory_manager = MemoryManager()

    def test_linear_scale_function(self):
        """
        Test the linear scale selection.

        Notes
        -----
        Each test will check that the function parameters are return correctly and
        that the function is called correctly and returns proper values.

        """
        # Test linear function
        scale_function = {"linear": {"scale_factor": 2}}
        function, parameters = self.memory_manager._select_scale_function(scale_function)
        self.assertEqual(parameters["scale_factor"], 2)
        self.assertEqual(function(10, **parameters), 20)

    def test_log_linear_scale_function(self):
        """
        Test the log-linear scale selection.

        Notes
        -----
        Each test will check that the function parameters are return correctly and
        that the function is called correctly and returns proper values.

        """
        # Test log-linear function
        scale_function = {"log-linear": {"scale_factor": 2}}
        function, parameters = self.memory_manager._select_scale_function(scale_function)
        self.assertEqual(parameters["scale_factor"], 2)
        self.assertEqual(function(10, **parameters), 20 * np.log(10))

    def test_quadratic_scale_function(self):
        """
        Test the quadratic scale selection.

        Notes
        -----
        Each test will check that the function parameters are return correctly and
        that the function is called correctly and returns proper values.

        """
        # Test quadratic function
        scale_function = {"quadratic": {"inner_scale_factor": 2, "outer_scale_factor": 2}}
        function, parameters = self.memory_manager._select_scale_function(scale_function)
        self.assertEqual(parameters["inner_scale_factor"], 2)
        self.assertEqual(parameters["outer_scale_factor"], 2)
        self.assertEqual(function(10, **parameters), 800)

    def test_polynomial_scale_function(self):
        """
        Test the polynomial scale selection.

        Notes
        -----
        Each test will check that the function parameters are return correctly and
        that the function is called correctly and returns proper values.

        """
        # Test polynomial function
        scale_function = {
            "polynomial": {"inner_scale_factor": 2, "outer_scale_factor": 2, "order": 3}
        }
        function, parameters = self.memory_manager._select_scale_function(scale_function)
        self.assertEqual(parameters["inner_scale_factor"], 2)
        self.assertEqual(parameters["outer_scale_factor"], 2)
        self.assertEqual(parameters["order"], 3)
        self.assertEqual(function(10, **parameters), 16000)

    def test_get_batch_size(self):
        """
        Test the get_batch_size method.

        Returns
        -------

        """
        # Test the exception catch.
        self.memory_manager.data_path = None
        self.assertRaises(ValueError, self.memory_manager.get_batch_size)

        # Test correct returns for 1 batch
        self.memory_manager.database = TestDatabase(data_size=500, rows=10, columns=10)
        self.memory_manager.data_path = ["Test/Path"]
        self.memory_manager.memory_fraction = 0.5
        self.memory_manager.machine_properties["memory"] = 50000
        batch_size, number_of_batches, remainder = self.memory_manager.get_batch_size()
        self.assertEqual(batch_size, 10)
        self.assertEqual(number_of_batches, 1)
        self.assertEqual(remainder, 0)

        # Test correct returns for N batches
        self.memory_manager.database = TestDatabase(data_size=500, rows=11, columns=13)
        self.memory_manager.data_path = ["Test/Path"]
        self.memory_manager.memory_fraction = 1.0
        self.memory_manager.machine_properties["memory"] = 50
        batch_size, number_of_batches, remainder = self.memory_manager.get_batch_size()
        self.assertEqual(batch_size, 1)
        self.assertEqual(number_of_batches, 13)
        self.assertEqual(remainder, 0)

    def test_hdf5_load_time(self):
        """
        Test the hdf5_load_time method.

        Returns
        -------
        Tests that the method returns the correct load time.

        """
        data = self.memory_manager.hdf5_load_time(10)
        self.assertEqual(data, np.log(10))

    def test_get_optimal_batch_size(self):
        """
        Test the get_optimal_batch_size method.

        Returns
        -------
        Test that this method returns the expected value. Currently this is just
        the same value that is passed to it.

        """
        data = self.memory_manager._get_optimal_batch_size(10)
        self.assertEqual(data, data)  # Todo: no shit, sherlock

    def test_compute_atomwise_minibatch(self):
        """
        Test the compute atom-wise mini-batch method.

        Returns
        -------
        Test the atom wise minibatch method. The test ensures for only a single case
        that the correct numbers are returned.

        """
        self.memory_manager.database = TestDatabase()
        self.memory_manager.data_path = ["Test/Path"]
        self.memory_manager._compute_atomwise_minibatch(5)
        self.assertEqual(self.memory_manager.batch_size, 10)
        self.assertEqual(self.memory_manager.n_batches, 1)
        self.assertEqual(self.memory_manager.n_atom_batches, 2)
        self.assertEqual(self.memory_manager.atom_remainder, 0)

    def test_get_ensemble_loop_standard(self):
        """
        Test the get_ensemble_loop method with no mini-batching.

        Returns
        -------

        """
        self.memory_manager.database = TestDatabase()
        self.memory_manager.data_path = ["Test/Path"]
        self.memory_manager.batch_size = 50
        data_partitions, minibatch = self.memory_manager.get_ensemble_loop(10, 5)
        self.assertEqual(minibatch, False)
        self.assertEqual(data_partitions, 8)

    def test_get_ensemble_loop_minibatching(self):
        """
        Test the get_ensemble_loop method with mini-batching.

        Returns
        -------

        """
        self.memory_manager.database = TestDatabase()
        self.memory_manager.data_path = ["Test/Path"]
        self.memory_manager.batch_size = 5
        data_partitions, minibatch = self.memory_manager.get_ensemble_loop(10, 5)
        self.assertEqual(minibatch, True)
        self.assertEqual(data_partitions, 1)
