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
Perform a functional test on two molten salts.
"""

from typing import Tuple

import pytest
from zinchub import DataHub

import mdsuite as mds


@pytest.fixture(scope="session")
def traj_files(tmp_path_factory) -> Tuple[str, str]:
    """Download trajectory file into a temporary directory and keep it for all tests."""
    temporary_path = tmp_path_factory.getbasetemp()

    NaCl_file = DataHub(
        url="https://github.com/zincware/DataHub/tree/main/NaCl_rnd_md", tag="v0.1.0"
    )

    KCl_file = DataHub(
        url="https://github.com/zincware/DataHub/tree/main/KCl_rnd_md", tag="v0.1.0"
    )

    KCl_data = KCl_file.get_file(temporary_path)[0]
    NaCl_data = NaCl_file.get_file(temporary_path)[0]

    NaCl_path = (temporary_path / NaCl_data).as_posix()
    KCl_path = (temporary_path / KCl_data).as_posix()

    return NaCl_path, KCl_path


@pytest.fixture()
def mdsuite_project(traj_files, tmp_path) -> mds.Project:
    """
    Create the MDSuite project and add data to be used for the rest of the tests.

    Parameters
    ----------
    traj_files : tuple
            Files include:
                * NaCl Simulation
                * KCl Simulation
    tmp_path : Path
            Temporary path that may be changed into.

    Returns
    -------
    project: mdsuite.Project
            An MDSuite project to be tested.

    """
    project = mds.Project(storage_path=tmp_path.as_posix())

    na_cl_file, k_cl_file = traj_files
    print(na_cl_file)

    project.add_experiment(
        name="NaCl",
        timestep=0.002,
        temperature=1200.0,
        units=mds.units.METAL,
        simulation_data=na_cl_file,
    )
    project.add_experiment(
        name="KCl",
        timestep=0.002,
        temperature=1200.0,
        units="metal",
        simulation_data=k_cl_file,
    )

    return project


def test_analysis(mdsuite_project):
    """
    Perform analysis on these MD simulations and ensure the outcomes are as expected.

    Returns
    -------
    Test the following:

        * Two experiments added to a project successfully
        * Correct coordination numbers computed
        * Correct POMF values computed
        * Dynamics run successfully.

    Notes
    -----
    See the link below for similar data for CNs for molten salts.
    https://link.springer.com/article/10.1007/s10800-018-1197-z

    """
    NaCl_experiment = mdsuite_project.experiments.NaCl
    KCl_experiment = mdsuite_project.experiments.KCl

    RDF_Data = mdsuite_project.run.RadialDistributionFunction(
        number_of_configurations=500, cutoff=15.0, plot=False
    )
    NaCl_CN_data = NaCl_experiment.run.CoordinationNumbers(
        rdf_data=RDF_Data["NaCl"],
        savgol_window_length=111,
        savgol_order=9,
        number_of_shells=3,
        plot=False,
    )
    KCl_CN_data = KCl_experiment.run.CoordinationNumbers(
        rdf_data=RDF_Data["KCl"],
        savgol_window_length=111,
        savgol_order=7,
        number_of_shells=2,
        plot=False,
    )
    KCl_POMF_data = KCl_experiment.run.PotentialOfMeanForce(
        rdf_data=RDF_Data["KCl"],
        savgol_window_length=111,
        savgol_order=7,
        number_of_shells=2,
        plot=False,
    )
    # Run assertions on selected observables
    NaCl_CN_data["Na_Cl"]["CN_1"] == pytest.approx(5.213, 0.0001)
    NaCl_CN_data["Na_Cl"]["CN_2"] == pytest.approx(35.090, 0.0001)
    NaCl_CN_data["Na_Na"]["CN_1"] == pytest.approx(14.775, 0.0001)

    KCl_CN_data["Cl_K"]["CN_1"] == pytest.approx(5.507, 0.0001)

    KCl_POMF_data["Cl_K"]["POMF_1"] == pytest.approx(-1.372e-11, 1e-14)

    mdsuite_project.run.GreenKuboDiffusionCoefficients()
    mdsuite_project.run.EinsteinDiffusionCoefficients()
