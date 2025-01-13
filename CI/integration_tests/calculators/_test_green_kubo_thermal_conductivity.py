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

import json
import os
from pathlib import Path

import numpy as np
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
    return NaCl.get_analysis(analysis="GreenKuboThermalConductivity.json")


def test_roject(traj_files, true_values, tmp_path):
    """Test the green_kubo_thermal_conductivity called from the project class."""
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment(
        "NaCl", simulation_data=traj_files[0], timestep=0.002, temperature=1400
    )

    project.run.GreenKuboThermalConductivity(plot=False)

    data_dict = project.load.GreenKuboThermalConductivity()[0].data_dict

    data = Path(r"calculators\data\green_kubo_thermal_conductivity.json")

    data.write_text(json.dumps(data_dict))

    np.testing.assert_array_almost_equal(data_dict["x"], true_values["x"])
    np.testing.assert_array_almost_equal(
        data_dict["uncertainty"], true_values["uncertainty"]
    )


def test_experiment(traj_files, true_values, tmp_path):
    """Test the green_kubo_thermal_conductivity called from the experiment class."""
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment(
        "NaCl", simulation_data=traj_files[0], timestep=0.002, temperature=1400
    )

    project.experiments["NaCl"].run.GreenKuboThermalConductivity(plot=False)

    data_dict = (
        project.experiments["NaCl"].load.GreenKuboThermalConductivity()[0].data_dict
    )

    np.testing.assert_array_almost_equal(data_dict["x"], true_values["x"])
    np.testing.assert_array_almost_equal(
        data_dict["uncertainty"], true_values["uncertainty"]
    )
