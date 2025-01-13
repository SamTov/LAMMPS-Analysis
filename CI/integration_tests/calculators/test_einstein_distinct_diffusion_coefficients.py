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
Test the einstein distinct diffusion coefficient module.

Notes
-----
Currently this test only checks that these calculators actually run it does not compare
values.

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
    return NaCl.get_analysis(analysis="RadialDistributionFunction.json")


@pytest.mark.parametrize("desired_memory", (None, 0.001))
def test_eddc_project(traj_file, true_values, tmp_path, desired_memory):
    """Test the EinsteinDistinctDiffusionCoefficients called from the project class."""
    with mds.utils.helpers.change_memory_fraction(desired_memory=desired_memory):
        os.chdir(tmp_path)
        project = mds.Project()
        project.add_experiment(
            "NaCl", simulation_data=traj_file, timestep=0.002, temperature=1400
        )

        project.run.EinsteinDistinctDiffusionCoefficients(
            plot=False, data_range=300, correlation_time=100
        )

    # data_dict = project.load.EinsteinDistinctDiffusionCoefficients()[0].data_dict
    #
    # data = Path(
    #     r"C:\Users\fabia\Nextcloud\DATA\JupyterProjects\MDSuite\CI\integration_tests"
    #     r"\calculators\data\einstein_distinct_diffusion_coefficients.json"
    # )
    #
    # data.write_text(json.dumps(data_dict))
    #
    # np.testing.assert_array_almost_equal(data_dict["x"], true_values["x"])
    # np.testing.assert_array_almost_equal(
    #     data_dict["uncertainty"], true_values["uncertainty"]
    # )


def test_eddc_experiment(traj_file, true_values, tmp_path):
    """Test the EinsteinDistinctDiffusionCoefficients called from the experiment class."""
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment(
        "NaCl", simulation_data=traj_file, timestep=0.002, temperature=1400
    )

    project.experiments["NaCl"].run.EinsteinDiffusionCoefficients(
        plot=False, data_range=300, correlation_time=100
    )
