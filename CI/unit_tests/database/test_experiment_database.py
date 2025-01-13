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
from zinchub import DataHub

import mdsuite as mds
import mdsuite.file_io.lammps_trajectory_files
from mdsuite.database.simulation_database import MoleculeInfo, SpeciesInfo


@pytest.fixture(scope="session")
def traj_file(tmp_path_factory) -> str:
    """Download trajectory file into a temporary directory and keep it for all tests."""
    temporary_path = tmp_path_factory.getbasetemp()

    NaCl = DataHub(
        url="https://github.com/zincware/DataHub/tree/main/NaCl_gk_i_q", tag="v0.1.0"
    )
    NaCl.get_file(path=temporary_path)

    return (temporary_path / NaCl.file_raw).as_posix()


def test_read_files(tmp_path, traj_file):
    """Test that read_files is saved correctly."""
    os.chdir(tmp_path)
    project_1 = mds.Project()
    file_proc = mdsuite.file_io.lammps_trajectory_files.LAMMPSTrajectoryFile(traj_file)
    project_1.add_experiment(name="Exp01", timestep=0.1)  # todo bad place for time step
    project_1.experiments["Exp01"].add_data(file_proc)
    assert len(project_1.experiments["Exp01"].read_files) == 1


def test_project_temperature(tmp_path):
    """Test that the project description is stored correctly in the database."""
    os.chdir(tmp_path)
    project_1 = mds.Project()
    project_1.add_experiment(name="Exp01")
    project_1.experiments["Exp01"].temperature = 9000

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    assert project_2.experiments["Exp01"].temperature == 9000


def test_project_time_step(tmp_path):
    """Test that the project description is stored correctly in the database."""
    os.chdir(tmp_path)
    project_1 = mds.Project()
    project_1.add_experiment(name="Exp01")
    project_1.experiments["Exp01"].time_step = 1

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    assert project_2.experiments["Exp01"].time_step == 1


def test_project_number_of_configurations(tmp_path):
    """Test that the project description is stored correctly in the database."""
    os.chdir(tmp_path)
    project_1 = mds.Project()
    project_1.add_experiment(name="Exp01")
    project_1.experiments["Exp01"].number_of_configurations = 100

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    assert project_2.experiments["Exp01"].number_of_configurations == 100


def test_project_number_of_atoms(tmp_path):
    """Test that the project description is stored correctly in the database."""
    os.chdir(tmp_path)
    project_1 = mds.Project()
    project_1.add_experiment(name="Exp01")
    project_1.experiments["Exp01"].number_of_atoms = 100

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    assert project_2.experiments["Exp01"].number_of_atoms == 100


def test_species(tmp_path):
    """Test that the species are stored correctly in the database."""
    os.chdir(tmp_path)
    species = {
        "H": SpeciesInfo(name="H", n_particles=3, mass=1, properties=[]),
        "Cl": SpeciesInfo(name="Cl", n_particles=3, mass=35.45, properties=[]),
    }

    project_1 = mds.Project()
    project_1.add_experiment(name="Exp01")
    project_1.experiments["Exp01"].species = species

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    assert project_2.experiments["Exp01"].species == species

    species_dict = {k: dataclasses.asdict(v) for k, v in species.items()}

    project_1 = mds.Project()
    project_1.add_experiment(name="Exp01")
    project_1.experiments["Exp01"].species = species_dict

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    assert project_2.experiments["Exp01"].species == species


def test_molecules(tmp_path):
    """Test that the molecules are stored correctly in the database."""
    os.chdir(tmp_path)

    molecule = {
        "H": MoleculeInfo(name="H", properties=[], mass=1, groups={}, n_particles=3),
        "Cl": MoleculeInfo(name="Cl", properties=[], mass=1, groups={}, n_particles=3),
    }

    project_1 = mds.Project()
    project_1.add_experiment(name="Exp01")
    project_1.experiments["Exp01"].molecules = molecule

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    assert project_2.experiments["Exp01"].molecules == molecule


def test_project_box_array(tmp_path):
    """Test that the project description is stored correctly in the database."""
    os.chdir(tmp_path)
    box_array = np.array([1.0, 1.414, 1.732])

    project_1 = mds.Project()
    project_1.add_experiment(name="Exp01")
    project_1.experiments["Exp01"].box_array = box_array

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    np.testing.assert_array_equal(project_2.experiments["Exp01"].box_array, box_array)


def test_experiment_simulation_data(tmp_path):
    """Test that the experiment simulation data is stored correctly in the database."""
    os.chdir(tmp_path)
    simulation_data = {
        "a_5": [10.0, 11.0, 12.0],
        "b_test": "HelloWorld",
        "c": 15.0,
    }  # can only handle float and str

    project_1 = mds.Project()
    project_1.add_experiment(name="Exp01")
    project_1.experiments["Exp01"].simulation_data = simulation_data

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")

    for key, val in project_2.experiments["Exp01"].simulation_data.items():
        assert val == simulation_data[key]


def test_experiment_simulation_data_nested(tmp_path):
    """Test that nested experiment simulation data is stored correctly in the database."""
    os.chdir(tmp_path)
    simulation_data = {"a": {"one": [1.0, 2.0, 3.0], "two": [4.0, 5.0, 6.0]}}

    project_1 = mds.Project()
    project_1.add_experiment(name="Exp01")
    project_1.experiments["Exp01"].simulation_data = simulation_data

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")

    for key, val in project_2.experiments["Exp01"].simulation_data.items():
        assert val == simulation_data[key]


def test_experiment_units(tmp_path):
    """Test that the experiment simulation data is stored correctly in the database."""
    from mdsuite.utils.units import Units

    os.chdir(tmp_path)
    custom_units = Units(
        time=1.0,
        length=1.0,
        energy=2.0,
        NkTV2p=1.0,
        temperature=100.0,
        pressure=123.0,
        boltzmann=25.0,
    )

    project_1 = mds.Project()
    project_1.add_experiment(name="Exp01", units=mds.units.SI)
    project_1.add_experiment(name="Exp02", units=custom_units)

    project_2 = mds.Project()

    assert project_2.experiments["Exp01"].units == mds.units.SI
    assert project_2.experiments["Exp02"].units == custom_units
