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

import numpy as np


def linear_scale_function(memory_usage: int, scale_factor: int = 1) -> int:
    """
    Apply a linear scaling to memory usage.

    Parameters
    ----------
    memory_usage : int
            naive memory usage, i.e., the exact memory of one configuration.
    scale_factor : int
            Scalar scaling factor for the memory usage in cases on non-size dependent
            inflation.

    Returns
    -------
    scaled_memory : int
            Amount of memory required per configuration loaded.

    """
    return memory_usage * scale_factor


def linearithmic_scale_function(memory_usage: int, scale_factor: int = 1) -> float:
    """
    Apply a linearithmic scaling to memory usage.

    Parameters
    ----------
    memory_usage : int
            naive memory usage, i.e., the exact memory of one configuration.
    scale_factor : int
            Scalar scaling factor for the memory usage in cases on non-size dependent
            inflation.

    Returns
    -------
    scaled_memory : float
            Amount of memory required per configuration loaded.

    """
    return scale_factor * memory_usage * np.log(memory_usage)


def quadratic_scale_function(
    memory_usage: int, inner_scale_factor: int = 1, outer_scale_factor: int = 1
) -> int:
    """
    Apply a quadratic scaling to memory usage.

    Parameters
    ----------
    memory_usage : int
            naive memory usage, i.e., the exact memory of one configuration.
    inner_scale_factor : int
            Scalar scaling factor for the inner multiplication
    outer_scale_factor : int
            Scalar scaling factor for the outer multiplication

    Returns
    -------
    scaled_memory : int
            Amount of memory required per configuration loaded.

    """
    return outer_scale_factor * (memory_usage * inner_scale_factor) ** 2


def polynomial_scale_function(
    memory_usage: int,
    inner_scale_factor: int = 1,
    outer_scale_factor: int = 1,
    order: int = 3,
) -> int:
    """
    Apply a polynomial scaling to memory usage.

    Parameters
    ----------
    memory_usage : int
            naive memory usage, i.e., the exact memory of one configuration.
    inner_scale_factor : int
            Scalar scaling factor for the inner multiplication
    outer_scale_factor : int
            Scalar scaling factor for the outer multiplication
    order : int
            Order of the polynomial.

    Returns
    -------
    scaled_memory : int
            Amount of memory required per configuration loaded.

    """
    return outer_scale_factor * (memory_usage * inner_scale_factor) ** order
