"""MDSuite: A Zincwarecode package.

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
Unit tests for the scaling functions module.
"""

import unittest

import numpy as np

from mdsuite.utils.scale_functions import (
    linear_scale_function,
    linearithmic_scale_function,
    polynomial_scale_function,
    quadratic_scale_function,
)


class TestScalingFunctions(unittest.TestCase):
    """A test class for the scaling functions module."""

    def test_linear_scaling_function(self):
        """
        Test the linear scaling function.

        Returns
        -------
        assert that the output is correct for several cases.

        """
        # Assert simple multiplication
        data = linear_scale_function(10, 1)
        self.assertEqual(data, 10)

        # Assert scaled multiplication
        data = linear_scale_function(10, 3)
        self.assertEqual(data, 30)

        # Assert array values
        data = linear_scale_function(np.array([5, 10, 20]), 2)
        np.testing.assert_array_equal(data, [10, 20, 40])

    def test_linearithmic_scaling_function(self):
        """
        Test the linearithmic scaling function.

        Returns
        -------
        assert that the output is correct for several cases.

        """
        # Assert simple multiplication
        data = linearithmic_scale_function(10, 1)
        self.assertAlmostEqual(data, 23.02585092994046, 5)

        # Assert scaled multiplication
        data = linearithmic_scale_function(10, 3)
        self.assertAlmostEqual(data, 69.07755278982138, 5)

        # Assert array values
        data = linearithmic_scale_function(np.array([5, 10, 20]), 2)
        np.testing.assert_almost_equal(data, [16.09437912, 46.05170186, 119.82929094])

    def test_quadratic_scaling_function(self):
        """
        Test the quadratic scaling function.

        Returns
        -------
        assert that the output is correct for several cases.

        """
        # Assert simple multiplication
        data = quadratic_scale_function(10, 1, 1)
        self.assertEqual(data, 100)

        # Assert scaled multiplication on inner
        data = quadratic_scale_function(10, 1, 3)
        self.assertEqual(data, 300)

        # Assert scaled multiplication on outer
        data = quadratic_scale_function(10, 2, 1)
        self.assertEqual(data, 400)

        # Assert array values
        data = quadratic_scale_function(np.array([5, 10, 20]), 1, 2)
        np.testing.assert_array_equal(data, [50, 200, 800])

    def test_polynomial_scaling_function(self):
        """
        Test the polynomial scaling function.

        Returns
        -------
        assert that the output is correct for several cases.

        """
        # Repeat quadratic test with poly api
        data = polynomial_scale_function(10, 1, 1, 2)
        self.assertEqual(data, 100)

        data = polynomial_scale_function(10, 1, 3, 2)
        self.assertEqual(data, 300)

        data = polynomial_scale_function(10, 2, 1, 2)
        self.assertEqual(data, 400)

        data = polynomial_scale_function(np.array([5, 10, 20]), 1, 2, 2)
        np.testing.assert_array_equal(data, [50, 200, 800])

        # Assert third order polynomial
        data = polynomial_scale_function(10, 1, 1, 3)
        self.assertEqual(data, 1000)

        # Assert fourth polynomial
        data = polynomial_scale_function(10, 1, 1, 4)
        self.assertEqual(data, 10000)
