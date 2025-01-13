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
Test the mdsuite testing modules.
"""

import time
import unittest

import numpy as np

from mdsuite.utils.testing import MDSuiteProcess, assertDeepAlmostEqual


class TestProcess(unittest.TestCase):
    """Test the process class."""

    def test_exception(self):
        """
        Test a raised exception.

        Returns
        -------

        """
        process = MDSuiteProcess(target=self._exception_throw)
        process.start()
        time.sleep(5)
        process.terminate()

        assert process.exception is not None

    def test_no_exception(self):
        """
        Test a raised exception.

        Returns
        -------

        """
        process = MDSuiteProcess(target=self._no_exception)
        process.start()
        time.sleep(5)
        process.terminate()

        assert process.exception is None

    @staticmethod
    def _exception_throw():
        """
        Process to throw exception.

        Returns
        -------
        -------.

        """
        raise ValueError("Test error")

    @staticmethod
    def _no_exception():
        """
        Process to not throw an exception.

        Returns
        -------
        -------.

        """
        pass


class TestAssertDeepAlmostEqual(unittest.TestCase):
    """Test the deep array assert testing method."""

    def test_almost_equal(self):
        """
        Test that arrays in dicts are found to be almost equal.

        Notes
        -----
        Taken from the module __main__

        """
        dict_2a = {"a": {"b": np.array([1, 2, 3, 4])}}
        dict_2b = {"a": {"b": [1, 2, 3, 4]}}
        dict_3a = {"a": {"c": np.array([1.10, 2.10, 3.11, 4.0])}}
        dict_3b = {"a": {"c": np.array([1.11, 2.09, 3.10, 4.0])}}

        assertDeepAlmostEqual(dict_3a, dict_3b, decimal=1)
        assertDeepAlmostEqual(dict_2a, dict_2b, decimal=1)
