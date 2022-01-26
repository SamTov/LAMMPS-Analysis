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
Functional test for the analysis of a GROMACS water simulation.
"""
from typing import List

import pytest
from zinchub import DataHub

import mdsuite as mds
import mdsuite.file_io.chemfiles_read
from mdsuite.utils import Units


@pytest.fixture(scope="session")
def traj_files(tmp_path_factory) -> List[str]:
    """Download trajectory file into a temporary directory and keep it for all tests"""
    temporary_path = tmp_path_factory.getbasetemp()

    water = DataHub(url="https://github.com/zincware/DataHub/tree/main/Water_14_Gromacs")
    water.get_file(temporary_path)
    file_paths = [(temporary_path / f).as_posix() for f in water.file_raw]
    return file_paths


@pytest.fixture()
def mdsuite_project(traj_files, tmp_path) -> mdsuite.Project:
    """
    Create the MDSuite project and add data to be used for the rest of the tests.

    Parameters
    ----------
    traj_files : list
            Files include:
                * Water Simulation
    tmp_path : Path
            Temporary path that may be changed into.

    Returns
    -------
    project: mdsuite.Project
            An MDSuite project to be tested.
    """
    gmx_units = Units(
        time=1e-12,
        length=1e-10,
        energy=1.6022e-19,
        NkTV2p=1.6021765e6,
        boltzmann=8.617343e-5,
        temperature=1,
        pressure=100000,
    )
    project = mds.Project(storage_path=tmp_path.as_posix())

    file_reader = mdsuite.file_io.chemfiles_read.ChemfilesRead(
        traj_file_path=traj_files[2], topol_file_path=traj_files[0]
    )

    project.add_experiment(
        name="water_sim",
        timestep=0.002,
        temperature=300.0,
        units=gmx_units,
        simulation_data=file_reader,
    )
    project.run.CoordinateUnwrapper()
    return project


def test_water_analysis(mdsuite_project):
    """Run a functional test by performing a study on an MD simulation of water"""

    water = mdsuite_project.experiments["water_sim"]

    water.run.MolecularMap(
        molecules={"water": {"smiles": "[H]O[H]", "amount": 14, "cutoff": 1.7}}
    )
    mdsuite_project.run.AngularDistributionFunction(plot=False)
    mdsuite_project.run.RadialDistributionFunction(plot=False)
    mdsuite_project.run.AngularDistributionFunction(plot=False, molecules=True)
    mdsuite_project.run.RadialDistributionFunction(plot=False, molecules=True)
    mdsuite_project.run.EinsteinDiffusionCoefficients(plot=False)
    mdsuite_project.run.EinsteinDiffusionCoefficients(plot=False, molecules=True)
    mdsuite_project.run.GreenKuboDiffusionCoefficients(plot=False)
