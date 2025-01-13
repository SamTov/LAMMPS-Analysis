"""
MDSuite: A zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the zincwarecode Project.

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
Module for testing the calculator helper methods.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_raises

from mdsuite.utils.calculator_helper_methods import (
    correlate,
    fit_einstein_curve,
    msd_operation,
)


class TestCalculatorHelperMethods:
    """Test suite for the calculator helper methods."""

    def test_fit_einstein_curve(self):
        """
        Test the fitting of a line.

        Notes
        -----
        Tests the following:

        * Returns correct gradient on a straight line
        * Returns correct gradient on a multi-regime line

        """
        x_data = np.linspace(0, 1000, 1000)

        # y = 5x + c
        y_data = 5 * x_data + 3

        popt, pcov, gradients, gradient_errors = fit_einstein_curve(
            x_data=x_data, y_data=y_data, fit_max_index=999
        )
        assert popt[0] == pytest.approx(5.0, 0.01)

        # exp(0.05x)x^2 + 5x + 3
        y_data = np.exp(-0.05 * x_data) * x_data**2 + 5 * x_data + 3

        popt, pcov, gradients, gradient_errors = fit_einstein_curve(
            x_data=x_data, y_data=y_data, fit_max_index=999
        )
        assert popt[0] == pytest.approx(5.0, 0.01)

    def test_correlate(self):
        """
        Test the correlate helper function.

        Returns
        -------
        Tests to see if the net cross correlation between a sine with itself and a sine
        with a lagged version of itself is zero.

        The first signal is auto-correlated, the second is perfectly anti-correlated.
        Therefore, when summed, they should cancel to zero.

        """
        # generate 10 points
        t = np.arange(10)
        # Create a 3d array
        x_data = np.vstack((t, t, t)).reshape(10, 3)

        sine_data = np.sin(x_data)
        lagged_sine_data = np.sin(x_data + np.pi)

        auto_correlation = np.array(correlate(sine_data, sine_data))
        cross_correlation = np.array(correlate(sine_data, lagged_sine_data))

        assert_raises(AssertionError, assert_array_equal, auto_correlation, np.zeros(10))
        assert_raises(AssertionError, assert_array_equal, cross_correlation, np.zeros(10))

        # Clip to correlate precision.
        summed_data = auto_correlation + cross_correlation
        summed_data[summed_data < 1e-10] = 0.0

        assert summed_data.sum() == 0.0

    def test_msd_operation(self):
        """
        Test the msd helper function.

        Returns
        -------
        Tests to see if the net cross correlation between a sine with itself and a sine
        with a lagged version of itself is zero.

        The first signal is auto-correlated, the second is perfectly anti-correlated.
        Therefore, when summed, they should cancel to zero.

        """
        # generate 10 points
        t = np.arange(10)
        # Create a 3d array
        x_data = np.vstack((t, t, t)).reshape(10, 3)

        sine_data = np.sin(x_data)
        lagged_sine_data = np.sin(x_data + np.pi)

        auto_correlation = np.array(msd_operation(sine_data, sine_data))
        cross_correlation = np.array(msd_operation(sine_data, lagged_sine_data))

        assert_raises(AssertionError, assert_array_equal, auto_correlation, np.zeros(10))
        assert_raises(AssertionError, assert_array_equal, cross_correlation, np.zeros(10))

        # Clip to correlate precision.
        summed_data = auto_correlation + cross_correlation
        summed_data[summed_data < 1e-10] = 0.0

        assert summed_data.sum() == 0.0
