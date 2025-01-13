"""MDSuite helpers.

This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0.

Copyright Contributors to the Zincware Project.

Description:
"""

import contextlib

import numpy as np

from mdsuite.utils.meta_functions import get_machine_properties


class NoneType:
    """Custom NoneType that is != None.

    Examples
    --------
    you can check for NoneType or None
    >>> x = NoneType
    >>> x is NoneType
    >>> x is not None

    """

    def __init__(self):
        raise NotImplementedError()


def compute_memory_fraction(desired_memory: float, total_memory: float = None):
    """
    Compute fraction of memory for the current system.

    Parameters
    ----------
    desired_memory : float
            Amount of memory in GB to be used.
    total_memory : float (default = None)
            If not None, this will be used as the total memory, good for testing.

    Returns
    -------
    memory fraction : float
            What fraction of the current systems memory this value corresponds to.
            If this number is above 1, it is clipped to 1

    """
    if total_memory is None:
        total_memory = get_machine_properties()["memory"] / (1024.0**3)

    memory_fraction = desired_memory / total_memory

    return np.clip(memory_fraction, None, 0.9)


@contextlib.contextmanager
def change_memory_fraction(desired_memory):
    """Context manager to adapt the memory within.

    Parameters
    ----------
    desired_memory: float
        Amount of memory in GB to be used.

    Yields
    ------
        environment where the 'config.memory_fraction' is adapted
        in regard to the desired_memory.

    """
    import mdsuite

    default = mdsuite.config.memory_fraction
    if desired_memory is not None:
        mdsuite.config.memory_fraction = mdsuite.utils.helpers.compute_memory_fraction(
            desired_memory
        )
    try:
        yield
    finally:
        mdsuite.config.memory_fraction = default
