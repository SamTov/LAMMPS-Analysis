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
    Check GK correctness by evaluating a system of noninteracting underdamped Langevin
    particles. Here, the diffusion coefficient as well as the velocity autocorrelation
    are exactly known.
    """
    with mds.utils.helpers.change_memory_fraction(desired_memory=desired_memory):
        diff_coeff = 1.2345
        kT = 0.7
        mass = 1.5
        time_step = 0.1
        n_part = 100

        gamma = kT / diff_coeff
        relaxation_time = mass / gamma
        n_step = int(100 * relaxation_time / time_step)
        vacf_range = int(7 * relaxation_time / time_step)

        tf.random.set_seed(19941003)
        random_force = np.sqrt(2 * kT * gamma / time_step) * tf.random.normal(
            shape=(n_step, n_part, 3), mean=0, stddev=1
        )
        # Euler integration of the Langevin equation.
        # Friction depends on velocity of the last step
        vel = np.zeros((n_step, n_part, 3))
        for i in range(1, n_step):
            force = random_force[i, :, :] - gamma * vel[i - 1, :, :]
            vel[i, :, :] = vel[i - 1, :, :] + time_step * force / mass

        os.chdir(tmp_path)
        project = mds.Project()
        # introduce nontrivial units to make sure all conversions are correct

        units = dataclasses.replace(mdsuite.units.SI, length=0.000007654, time=0.0056789)

        exp = project.add_experiment(
            "test_diff_coeff", timestep=time_step, temperature=kT, units=units
        )

        vel_prop = mdsuite_properties.velocities
        species = SpeciesInfo(
            name="test_species", n_particles=n_part, properties=[vel_prop], mass=mass
        )
        metadata = TrajectoryMetadata(
            species_list=[species],
            n_configurations=n_step,
            sample_rate=1,
        )
        data = TrajectoryChunkData(species_list=[species], chunk_size=n_step)
        data.add_data(vel, 0, species.name, vel_prop.name)
        proc = ScriptInput(data=data, metadata=metadata, name="test_name")
        exp.add_data(proc)

        res = exp.run.GreenKuboDiffusionCoefficients(data_range=vacf_range, plot=False)[
            species.name
        ]

        time_should_be = time_step * np.arange(0, vacf_range) * units.time
        thermal_vel_SI = np.sqrt(3 * kT / mass) * units.length / units.time
        relaxation_time_SI = relaxation_time * units.time
        vacf_should_be = thermal_vel_SI**2 * np.exp(-time_should_be / relaxation_time_SI)
        diff_coeff_should_be = diff_coeff * units.length**2 / units.time

        np.testing.assert_allclose(res["time"], time_should_be, atol=1e-6)
        np.testing.assert_allclose(
            res["acf"], vacf_should_be, atol=0.03 * vacf_should_be[0]
        )
        np.testing.assert_allclose(
            res["diffusion_coefficient"], diff_coeff_should_be, rtol=3e-2
        )
