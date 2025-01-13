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
"""

import os

import numpy as np
import pytest

import mdsuite as mds
import mdsuite.file_io.script_input
import mdsuite.transformations.test_trafos
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.database.simulation_database import (
    PropertyInfo,
    SpeciesInfo,
    TrajectoryChunkData,
    TrajectoryMetadata,
)


def load_pos_into_exp(exp, sp_name="test_species", unwrapped=True):
    n_config = 100
    n_part = 10

    if unwrapped:
        pos_prop = (
            mdsuite.database.mdsuite_properties.mdsuite_properties.unwrapped_positions
        )
    else:
        pos_prop = mdsuite.database.mdsuite_properties.mdsuite_properties.positions
    species = SpeciesInfo(sp_name, n_part, [pos_prop])
    mdata = TrajectoryMetadata(n_config, [species])

    pos = np.random.random((n_config, n_part, pos_prop.n_dims))

    if not unwrapped:
        pos /= np.max(pos)

    chunk = TrajectoryChunkData([species], n_config)
    chunk.add_data(pos, 0, species.name, pos_prop.name)

    np_read = mdsuite.file_io.script_input.ScriptInput(chunk, mdata, "test_reader")
    exp.add_data(np_read)

    return species, pos


def test_automatic_coordinate_unwrapping(tmp_path):
    """
    Check if positions are automatically unwrapped if needed in another transformation.
    This also tests the fallback mechanism in case the first trafo (unwrap_via_indices)
    does not work.
    """
    os.chdir(tmp_path)

    n_part = 10
    t_step = 0.123
    box_l = [1.1, 2.2, 3.3]

    project = mds.Project()
    project.add_experiment(name="TestExp", timestep=t_step)
    exp = project.experiments["TestExp"]
    exp.box_array = box_l
    species, _ = load_pos_into_exp(exp, unwrapped=False)

    exp.run.VelocityFromPositions()
    unwrapped_pos = exp.load_matrix(
        property_name=mdsuite_properties.unwrapped_positions.name, species=[species.name]
    )[f"{species.name}/{mdsuite_properties.unwrapped_positions.name}"]

    assert len(unwrapped_pos) == n_part


def test_full_transformation_with_values(tmp_path):
    """
    Check for one transformation, that the correct data is loaded and
    transferred to the actual transformation function.
    """
    os.chdir(tmp_path)

    t_step = 0.123
    project = mds.Project()
    project.add_experiment(name="TestExp", timestep=t_step)
    exp = project.experiments["TestExp"]

    species, pos = load_pos_into_exp(exp)

    exp.run.VelocityFromPositions()
    vels_mds = exp.load_matrix(
        property_name="Velocities_From_Positions", species=[species.name]
    )[f"{species.name}/Velocities_From_Positions"]

    pos = np.swapaxes(pos, 0, 1)
    vels_numpy = (pos[:, 1:, :] - pos[:, :-1, :]) / t_step
    last_vels = vels_numpy[:, -1, :]
    vels_numpy = np.concatenate((vels_numpy, last_vels[:, None, :]), axis=1)

    np.testing.assert_almost_equal(vels_mds, vels_numpy, decimal=4)


def test_not_found_errors(tmp_path):
    """Test that the correct error is thrown if input data cannot be found."""
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment(name="TestExp1234", timestep=12345)
    exp = project.experiments["TestExp1234"]
    load_pos_into_exp(exp)

    def check_madeup_prop(trafo_class, prop_name):
        trafo = trafo_class(
            input_properties=[PropertyInfo(name="MadeUpProperty", n_dims=42)],
            output_property=PropertyInfo(name=prop_name, n_dims=2),
        )
        with pytest.raises(
            mdsuite.transformations.transformations.CannotFindPropertyError
        ):
            exp.cls_transformation_run(trafo)

    check_madeup_prop(mdsuite.transformations.test_trafos.TestSingleSpecies, "test1")
    check_madeup_prop(mdsuite.transformations.test_trafos.TestMultispecies, "test2")


def test_save_to_correct_name(tmp_path):
    """
    Check that the transformation result is at the correct place
    in the simulation database.
    """
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment(name="TestExp", timestep=12345)
    exp = project.experiments["TestExp"]

    species, _ = load_pos_into_exp(exp)

    trafo = mdsuite.transformations.test_trafos.TestSingleSpecies(
        input_properties=[mdsuite_properties.unwrapped_positions],
        output_property=PropertyInfo(name="test_single", n_dims=2),
    )
    exp.cls_transformation_run(trafo)

    ret = exp.load_matrix(species=[species.name], property_name="test_single")
    assert isinstance(ret, dict)

    trafo2 = mdsuite.transformations.test_trafos.TestMultispecies(
        input_properties=[mdsuite_properties.unwrapped_positions],
        output_property=PropertyInfo(name="test_multi", n_dims=2),
    )
    exp.cls_transformation_run(trafo2)
    ret2 = exp.load_matrix(
        species=[mds.utils.DatasetKeys.OBSERVABLES], property_name="test_multi"
    )
    assert isinstance(ret2, dict)


def test_data_from_species_and_experiment(tmp_path):
    """
    Test trafo that takes positions time dependent, charge from the species,
    and box_l from the experiment.
    """
    os.chdir(tmp_path)

    t_step = 0.123
    box_l = [1.1, 2.2, 3.3]

    project = mds.Project()
    project.add_experiment(name="TestExp", timestep=t_step)
    exp = project.experiments["TestExp"]

    species, _ = load_pos_into_exp(exp)
    exp.box_array = box_l
    exp.species[species.name].charge = 1.23435

    def check_trafo(trafo_class, prop_name):
        trafo = trafo_class(
            input_properties=[
                mdsuite_properties.unwrapped_positions,
                mdsuite_properties.charge,
                mdsuite_properties.box_length,
            ],
            output_property=PropertyInfo(name=prop_name, n_dims=2),
        )
        exp.cls_transformation_run(trafo)

    check_trafo(mdsuite.transformations.test_trafos.TestSingleSpecies, "test1")
    check_trafo(mdsuite.transformations.test_trafos.TestMultispecies, "test2")


def test_transformation_on_new_data_(tmp_path):
    """Check that after adding new data, the transformation still works."""
    os.chdir(tmp_path)
    project = mds.Project()

    def check_trafo(trafo_class, exp_name):
        project.add_experiment(name=exp_name, timestep=12345)
        exp = project.experiments[exp_name]

        trafo = trafo_class(
            input_properties=[mdsuite_properties.unwrapped_positions],
            output_property=PropertyInfo(name="test_prop", n_dims=2),
        )

        load_pos_into_exp(exp)
        exp.cls_transformation_run(trafo)
        load_pos_into_exp(exp)
        exp.cls_transformation_run(trafo)

    check_trafo(mdsuite.transformations.test_trafos.TestSingleSpecies, "test1")
    check_trafo(mdsuite.transformations.test_trafos.TestMultispecies, "test2")
