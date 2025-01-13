"""MDSuite Test for transformations and run options.

This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0.

Copyright Contributors to the Zincware Project.

Description:
"""

import pytest
from zinchub import DataHub

import mdsuite
import mdsuite.transformations


@pytest.fixture(scope="session")
def traj_file(tmp_path_factory) -> str:
    """Download trajectory file into a temporary directory and keep it for all tests."""
    temporary_path = tmp_path_factory.getbasetemp()

    NaCl = DataHub(
        url="https://github.com/zincware/DataHub/tree/main/NaCl_gk_i_q", tag="v0.1.0"
    )
    NaCl.get_file(path=temporary_path)

    return (temporary_path / NaCl.file_raw).as_posix()


@pytest.fixture()
def mdsuite_project(traj_file, tmp_path) -> mdsuite.Project:
    project = mdsuite.Project(storage_path=tmp_path.as_posix())
    project.add_experiment("NaCl", simulation_data=traj_file)

    return project


def test_from_project(mdsuite_project):
    """
    Test the unwrapping call from the project class.

    Notes
    -----
    Does not check actual values just runs the transformation.

    """
    mdsuite_project.run.CoordinateUnwrapper()


def test_from_project_twice(mdsuite_project):
    """
    Test the unwrapping call from the project class twice to ensure that it prevents
    attempted creation of a database group twice.

    Notes
    -----
    Does not check actual values just runs the transformation.

    """
    mdsuite_project.run.CoordinateUnwrapper()
    mdsuite_project.run.CoordinateUnwrapper()


def test_from_experiment(mdsuite_project):
    """
    Test the unwrapping call from the experiment class.

    Notes
    -----
    Does not check actual values just runs the transformation.

    """
    mdsuite_project.experiments.NaCl.run.CoordinateUnwrapper()


def test_from_experiment_twice(mdsuite_project):
    """
    Test the unwrapping call from the experiment class twice to ensure that it prevents
    attempted creation of a database group twice.

    Notes
    -----
    Does not check actual values just runs the transformation.

    """
    mdsuite_project.experiments.NaCl.run.CoordinateUnwrapper()
    mdsuite_project.experiments.NaCl.run.CoordinateUnwrapper()


def test_pass_instance_to_exp(mdsuite_project):
    """Test passing the transformation to the experiment class."""
    mdsuite_project.experiments.NaCl.cls_transformation_run(
        mdsuite.transformations.CoordinateUnwrapper()
    )


def test_call_with_instance(mdsuite_project):
    """Test instanciating and then calling the trafo."""
    trafo = mdsuite.transformations.CoordinateUnwrapper()
    mdsuite_project.experiments.NaCl.cls_transformation_run(trafo)
