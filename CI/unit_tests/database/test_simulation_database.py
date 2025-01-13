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
Test for module for the simulation database.
"""

import os
import tempfile
import unittest

import h5py as hf
import numpy as np

from mdsuite.database.simulation_database import Database


class TestScalingFunctions(unittest.TestCase):
    """A test class for the simulation database class."""

    def test_build_path_input(self):
        """
        Test the build path input method.

        Returns
        -------
        Asserts that the correct path is generated for a given input.

        """
        temp_dir = tempfile.TemporaryDirectory()
        os.chdir(temp_dir.name)
        database = Database()
        assertion = {
            "Na/Forces": (200, 5000, 3),
            "Pressure": (5000, 6),
            "Temperature": (5000, 1),
        }
        architecture = database._build_path_input(
            {
                "Na": {"Forces": (200, 5000, 3)},
                "Pressure": (5000, 6),
                "Temperature": (5000, 1),
            }
        )
        self.assertDictEqual(assertion, architecture)
        os.chdir("..")
        temp_dir.cleanup()

    def test_add_dataset(self):
        """
        Test the add_dataset method.

        Returns
        -------
        Assert that a dataset of the correct size is built.

        """
        temp_dir = tempfile.TemporaryDirectory()
        os.chdir(temp_dir.name)
        database = Database()
        database.initialize_database({"Na": {"Forces": (200, 5000, 3)}})
        with hf.File("database") as db:
            keys_top = list(db.keys())
            keys_bottom = list(db[keys_top[0]])
            ds_shape = db[f"{keys_top[0]}/{keys_bottom[0]}"].shape

        np.testing.assert_array_equal(keys_top, ["Na"])
        np.testing.assert_array_equal(keys_bottom, ["Forces"])
        np.testing.assert_equal(ds_shape, (200, 5000, 3))
        os.chdir("..")
        temp_dir.cleanup()

    def test_resize_array(self):
        """
        Test the resizing of a dataset.

        Returns
        -------
        Resizes a built dataset and checks that the size is now correct.

        """
        temp_dir = tempfile.TemporaryDirectory()
        os.chdir(temp_dir.name)
        database = Database()
        database.initialize_database({"Na": {"Forces": (200, 5000, 3)}})
        database.resize_datasets({"Na": {"Forces": (200, 300, 3)}})

        with hf.File("database") as db:
            keys_top = list(db.keys())
            keys_bottom = list(db[keys_top[0]])
            ds_shape = db[f"{keys_top[0]}/{keys_bottom[0]}"].shape

        np.testing.assert_array_equal(keys_top, ["Na"])
        np.testing.assert_array_equal(keys_bottom, ["Forces"])
        np.testing.assert_equal(ds_shape, (200, 5300, 3))
        os.chdir("..")
        temp_dir.cleanup()

    def test_database_exists(self):
        """
        Test the database_exists method.

        Returns
        -------
        Checks for a False and then True result.

        """
        temp_dir = tempfile.TemporaryDirectory()
        os.chdir(temp_dir.name)
        database = Database()
        assert not database.database_exists()
        database.initialize_database({"Na": {"Forces": (200, 5000, 3)}})
        assert database.database_exists()
        os.chdir("..")
        temp_dir.cleanup()

    def test_check_existence(self):
        """
        Test the check_existence method.

        Returns
        -------
        Checks for a True and False result.

        """
        temp_dir = tempfile.TemporaryDirectory()
        os.chdir(temp_dir.name)
        database = Database()
        database.initialize_database({"Na": {"Forces": (200, 5000, 3)}})
        assert not database.check_existence("Na/Positions")
        assert database.check_existence("Na/Forces")
        os.chdir("..")
        temp_dir.cleanup()
