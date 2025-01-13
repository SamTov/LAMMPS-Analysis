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
Test the meta functions module.
"""

import os

import numpy as np
import tensorflow as tf

from mdsuite.utils.meta_functions import (
    check_a_in_b,
    find_item,
    get_dimensionality,
    get_machine_properties,
    get_nearest_divisor,
    golden_section_search,
    gpu_available,
    join_path,
    line_counter,
    linear_fitting_function,
    optimize_batch_size,
    simple_file_read,
    split_array,
)


class TestMetaFunction:
    """A test class for the meta functions module."""

    def test_join_path(self):
        """
        Test the join_path method.

        Returns
        -------
        assert that join_path('a', 'b') is 'a/b'

        """
        assert join_path("a", "b") == "a/b"

    def test_get_dimensionality(self):
        """
        Test the get_dimensionality method.

        Returns
        -------
        assert that for all choices of dimension array the correct dimension comes out.

        """
        one_d = [1, 0, 0]
        two_d = [1, 1, 0]
        three_d = [1, 1, 1]
        assert get_dimensionality(one_d) == 1
        assert get_dimensionality(two_d) == 2
        assert get_dimensionality(three_d) == 3

    def test_get_machine_properties(self):
        """
        Test the get_machine_properties method.

        Returns
        -------
        This test will just run the method and check for a failure.

        """
        get_machine_properties()

    def test_line_counter(self):
        """
        Test the line_counter method.

        Returns
        -------
        Check that the correct number of lines is return for a test file.

        """
        data = [["ayy"], ["bee"], ["cee"], ["dee"]]
        name = "line_counter_test.txt"
        with open(name, "w") as f:
            for item in data:
                f.write(f"{item[0]}\n")
        assert line_counter(name) == 4
        os.remove(name)

    def test_optimize_batch_size(self):
        """
        Test the optimize_batch_size method.

        Returns
        -------
        assert the correct batch size is returned for several inputs.

        """
        # Assert that the batch number is the full trajectory.
        number_of_configurations = 10
        _file_size = 100
        _memory = 1000000
        batch_size = optimize_batch_size(
            "None", number_of_configurations, _file_size, _memory, test=True
        )
        assert batch_size == 10

        # Assert that the batch number is the half the trajectory.
        number_of_configurations = 10
        _file_size = 100
        _memory = 250
        batch_size = optimize_batch_size(
            "None", number_of_configurations, _file_size, _memory, test=True
        )
        assert batch_size == 5

    def test_linear_fitting_function(self):
        """
        Test the linear_fitting_function method.

        Returns
        -------
        Assert the correction function values come out.

        """
        a = 5
        b = 3
        x = np.array([1, 2, 3, 4, 5])
        reference = a * x + b
        assert np.array_equal(linear_fitting_function(x, a, b), reference)

    def test_simple_file_read(self):
        """
        Test the simple_file_read method.

        Returns
        -------
        Assert that the arrays read in are as expected.

        """
        data = [["ayy"], ["bee"], ["cee"], ["dee"]]
        name = "line_counter_test.txt"
        with open(name, "w") as f:
            for item in data:
                f.write(f"{item[0]}\n")
        np.array_equal(simple_file_read(name), data)
        os.remove(name)

    def test_golden_section_search(self):
        """
        Test the golden_section_search method.

        Returns
        -------
        Asserts that the correct minimum is found.

        """

        def func(x: np.ndarray):
            """
            Test function.

            Parameters
            ----------
            x : np.ndarray.

            Returns
            -------
            x**2

            """
            return x**2

        x_dat = np.linspace(-10, 10, 1000)
        data = [x_dat, func(x_dat)]
        output = golden_section_search(data, 10, -10)
        assert output[0] == 0.010010010010010006
        assert output[1] == 0.03003003003003002

    def test_round_down(self):
        """
        Test the round_down method.

        Returns
        -------
        Assert the correct rounding occurs.

        """
        b = 10
        a = 9

        assert get_nearest_divisor(a, b) == 5

    def test_split_arrays(self):
        """
        Test the split_arrays method.

        Returns
        -------
        assert that array splitting has been performed correctly.

        """
        a = np.array([1, 2, 3, 10, 20, 30])
        assert np.array_equal(
            split_array(a, a < 10), [np.array([1, 2, 3]), np.array([10, 20, 30])]
        )

    def test_find_item(self):
        """
        Test the find item method.

        Returns
        -------
        assert that a deep item is retrieved from a dictionary.

        """
        test_1 = {"a": 4}  # test the first if statement
        test_2 = {
            "a": {"ae": {"aee": 1}},
            "b": {"be": {"bee": 4}},
        }  # test the case when the if statement fails.

        assert find_item(test_1, "a") == 4
        assert find_item(test_2, "aee") == 1

    def test_gpu_available(self):
        """
        Ideally this should be checked against something else than tf.config
        but I don't have any better measure.
        """
        assert gpu_available() == (len(tf.config.list_physical_devices("GPU")) > 1)

    def test_check_a_in_b(self):
        """
        Test that the a_in_b method works correctly.

        Returns
        -------
        Tests the following:

        * If a is in b a True is returned
        * If a is not in b a False is returned.

        """
        b = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        a = tf.constant([4, 5, 6])
        c = tf.constant([10, 11, 12])

        assert check_a_in_b(a, b)
        assert not check_a_in_b(c, b)
