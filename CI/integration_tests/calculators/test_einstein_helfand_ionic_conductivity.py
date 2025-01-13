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

import os

import pytest
from zinchub import DataHub

import mdsuite as mds


@pytest.fixture(scope="session")
def traj_file(tmp_path_factory) -> str:
    """Download trajectory file into a temporary directory and keep it for all tests."""
    temporary_path = tmp_path_factory.getbasetemp()

    NaCl = DataHub(
        url="https://github.com/zincware/DataHub/tree/main/NaCl_gk_i_q", tag="v0.1.0"
    )
    NaCl.get_file(path=temporary_path)

    return (temporary_path / NaCl.file_raw).as_posix()


@pytest.fixture(scope="session")
def true_values() -> dict:
    """Example fixture for downloading analysis results from github."""
    NaCl = DataHub(
        url="https://github.com/zincware/DataHub/tree/main/NaCl_gk_i_q", tag="v0.1.0"
    )
    return NaCl.get_analysis(analysis="EinsteinHelfandIonicConductivity.json")


@pytest.mark.parametrize("desired_memory", (None, 0.001))
def test_project(traj_file, true_values, tmp_path, desired_memory):
    """Test the Einstein_Helfand_Ionic_Conductivity called from the project class.

    Notes
    -----
    Test uncertainty is very high!

    """
    with mds.utils.helpers.change_memory_fraction(desired_memory=desired_memory):
        os.chdir(tmp_path)
        project = mds.Project()
        project.add_experiment(
            "NaCl", simulation_data=traj_file, timestep=0.002, temperature=1400
        )
        _ = project.run.EinsteinHelfandIonicConductivity(plot=False)
