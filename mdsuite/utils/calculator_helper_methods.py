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
Static methods used in calculators are kept here rather than polluting the parent class.
"""

import logging
from typing import Any, Iterable, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from numpy import ndarray
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

log = logging.getLogger(__name__)


def fit_einstein_curve(
    x_data: np.ndarray, y_data: np.ndarray, fit_max_index: int
) -> Tuple[Union[ndarray, Iterable, int, float], Any, list, list]:
    """
    Fit operation for Einstein calculations.

    Parameters
    ----------
    x_data : np.ndarray
            x data to use in the fitting.
    y_data : np.ndarray
            y_data to use in the fitting.
    fit_max_index : int
            Range at which to store values.

    Returns
    -------
    popt : list
            List of fit values
    pcov : list
            Covariance matrix of the fit values.

    """
    # Defined here for completeness.
    popt = []
    pcov = []

    def func(x, m, a):
        """
        Standard linear function for fitting.

        Parameters
        ----------
        x : list/np.array
                x axis tensor_values for the fit
        m : float
                gradient of the line
        a : float
                scalar offset, also the y-intercept for those who did not
                get much maths in school.

        Returns
        -------
        m * x + a

        """
        return m * x + a

    spline_data = UnivariateSpline(x_data, y_data, s=0, k=4)

    derivatives = spline_data.derivative(n=2)(x_data)

    derivatives[abs(derivatives) < 1e-5] = 0
    start_index = np.argmin(abs(derivatives))
    gradients = []
    gradient_errors = []

    for i in range(start_index + 2, len(y_data)):
        popt_temp, pcov_temp = curve_fit(
            func, xdata=x_data[start_index:i], ydata=y_data[start_index:i]
        )
        gradients.append(popt_temp[0])
        gradient_errors.append(np.sqrt(np.diag(pcov_temp))[0])

        if i == fit_max_index:
            popt = popt_temp
            pcov = pcov_temp

    return popt, pcov, gradients, gradient_errors


def correlate(ds_a: np.ndarray, ds_b: np.ndarray) -> np.ndarray:
    """
    Compute a simple correlation computation mapped over the spatial dimension of
    the array.

    Parameters
    ----------
    ds_a : np.ndarray (n_configurations, dimension)
            Tensor of the first set of data for a single particle.
    ds_b : np.ndarray (n_configurations, dimension)
            Tensor of the second set of data for a single particle.

    Returns
    -------
    Computes the correlation between the two data sets and averages over the spatial
    dimension.

    """

    def _correlate_op(a: np.ndarray, b: np.ndarray):
        """
        Actual correlation op to be mapped over the spatial dimension.

        Parameters
        ----------
        a : np.ndarray (n_configurations, dimension)
            Tensor of the first set of data for a single particle.
        b : np.ndarray (n_configurations, dimension)
            Tensor of the second set of data for a single particle.

        Returns
        -------
        correlation over a single dimension.

        """
        return jnp.correlate(a, b, mode="full")

    # We want to vmap over the last axis
    correlate_vmap = jax.vmap(_correlate_op, in_axes=-1)

    acf = np.mean(correlate_vmap(ds_a, ds_b), axis=0)

    return acf[int(len(acf) / 2) :]


def msd_operation(ds_a, ds_b) -> np.ndarray:
    """
    Perform an msd operation between two data sets mapping over spatial dimension.

    Parameters
    ----------
    ds_a : np.ndarray (n_timesteps, dimension)
    ds_b : np.ndarray (n_timesteps, dimension)

    Returns
    -------

    """

    def _difference_op(a, b):
        """
        Difference operation to map over spatial dimension.

        Parameters
        ----------
        a : (n_timesteps)
        b : (n_timesteps)

        Returns
        -------

        """
        return (a - a[0]) * (b - b[0])

    difference_vmap = jax.vmap(_difference_op, in_axes=-1)

    return np.mean(difference_vmap(ds_a, ds_b), axis=0)
