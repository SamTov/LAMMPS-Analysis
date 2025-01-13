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

import dataclasses
import os

import numpy as np
import pytest
import tensorflow as tf

import mdsuite as mds
import mdsuite.utils.units
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.database.simulation_database import (
    SpeciesInfo,
    TrajectoryChunkData,
    TrajectoryMetadata,
)
from mdsuite.file_io.script_input import ScriptInput


@pytest.mark.parametrize("desired_memory", (None, 0.001))
def test_calculator(tmp_path, desired_memory):
    """
    Check correctness of the msd and diffusion coefficient by generating data
    where these quantities are known.
    """
    with mds.utils.helpers.change_memory_fraction(desired_memory=desired_memory):
        diff_coeff = 1.2345
        time_step = 0.1

        n_part = 500
        n_step = 5000
        msd_range = 500
        msd_range = 200

        vel = np.sqrt(2 * diff_coeff / time_step) * tf.random.normal(
            shape=(n_step, n_part, 3), mean=0, stddev=1
        )
        pos = time_step * tf.math.cumsum(vel, axis=0)

        os.chdir(tmp_path)
        project = mds.Project()
        # introduce nontrivial units to make sure all conversions are correct
        units = dataclasses.replace(mdsuite.units.SI, length=0.5, time=1.3)
        exp = project.add_experiment(
            "test_diff_coeff", timestep=time_step, temperature=4.321, units=units
        )

        pos_prop = mdsuite_properties.unwrapped_positions
        species = SpeciesInfo(
            name="test_species", n_particles=n_part, properties=[pos_prop]
        )
        metadata = TrajectoryMetadata(
            species_list=[species],
            n_configurations=n_step,
            sample_rate=1,
        )
        data = TrajectoryChunkData(species_list=[species], chunk_size=n_step)
        data.add_data(pos, 0, species.name, pos_prop.name)
        proc = ScriptInput(data=data, metadata=metadata, name="test_name")
        exp.add_data(proc)

        res = exp.run.EinsteinDiffusionCoefficients(
            plot=False, correlation_time=1, data_range=msd_range
        )[species.name]

        time_should_be = time_step * np.arange(0, msd_range) * units.time
        diff_coeff_should_be = diff_coeff * units.length**2 / units.time
        msd_shouldbe = 6 * diff_coeff_should_be * time_should_be

        np.testing.assert_allclose(res["time"], time_should_be, atol=1e-5)
        np.testing.assert_allclose(res["msd"], msd_shouldbe, rtol=9e-1)
        np.testing.assert_allclose(
            res["diffusion_coefficient"], diff_coeff_should_be, rtol=2e-1
        )
