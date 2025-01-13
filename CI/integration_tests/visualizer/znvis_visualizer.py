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
Test that the visualizer runs.
"""

import os
import tempfile
import time
import unittest

from zinchub import DataHub

import mdsuite as mds
from mdsuite.utils.testing import MDSuiteProcess


class TestZnvisVisualizer(unittest.TestCase):
    """A test class for the Particle class."""

    def test_run(self):
        """
        Run the simple spheres tutorial and ensure it does not throw
        exceptions.

        Returns
        -------

        Notes
        -----
        TODO: We can set up a means of having the main thread check if the test is
            finished on the deployed process and then kill it. That will ensure that the
            test isn't prematurely closed and passes when it should have failed. For now
            120 seconds is more than enough time for this test but we can think of some
            improvements later.

        """
        process = MDSuiteProcess(target=self._run_app)
        process.start()
        time.sleep(90)  # give it 90 seconds to run.
        process.terminate()
        if process.exception:
            error, traceback = process.exception
            raise Exception(traceback)

        assert process.exception is None

    @staticmethod
    def _run_app():
        """
        Run the app to see that it works.

        Returns
        -------

        """
        temp_dir = tempfile.TemporaryDirectory()
        os.chdir(temp_dir.name)
        argon = DataHub(
            url="https://github.com/zincware/DataHub/tree/main/Ar/Ar_dft", tag="v0.1.0"
        )
        argon.get_file(path=".")
        project = mds.Project("Ar_test")
        project.add_experiment(
            name="argon",
            timestep=0.1,
            temperature=85.0,
            units="metal",
            simulation_data="Ar_dft_short.extxyz",
        )
        # project.experiments.argon.run_visualization() # has been removed

        os.chdir("..")
        temp_dir.cleanup()
