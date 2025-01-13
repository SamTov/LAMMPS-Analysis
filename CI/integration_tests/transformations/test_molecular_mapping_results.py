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
Test the outcome of molecular mapping.
"""

from typing import List, Tuple

import pytest
from zinchub import DataHub

import mdsuite
import mdsuite.file_io.chemfiles_read
import mdsuite.transformations
from mdsuite.database.simulation_database import MoleculeInfo
from mdsuite.utils import Units


@pytest.fixture(scope="session")
def traj_files(tmp_path_factory) -> Tuple[List[str], str]:
    """Download trajectory file into a temporary directory and keep it for all tests."""
    temporary_path = tmp_path_factory.getbasetemp()

    water = DataHub(
        url="https://github.com/zincware/DataHub/tree/main/Water_14_Gromacs", tag="v0.1.0"
    )
    water.get_file(temporary_path)
    file_paths = [(temporary_path / f).as_posix() for f in water.file_raw]

    bmim_bf4 = DataHub(
        url="https://github.com/zincware/DataHub/tree/main/Bmim_BF4", tag="v0.1.0"
    )
    bmim_bf4.get_file(path=temporary_path)

    return file_paths, (temporary_path / bmim_bf4.file_raw).as_posix()


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
    water_files = traj_files[0]
    bmim_file = traj_files[1]

    gmx_units = Units(
        time=1e-12,
        length=1e-10,
        energy=1.6022e-19,
        NkTV2p=1.6021765e6,
        boltzmann=8.617343e-5,
        temperature=1,
        pressure=100000,
    )
    project = mdsuite.Project(storage_path=tmp_path.as_posix())

    file_reader_1 = mdsuite.file_io.chemfiles_read.ChemfilesRead(
        traj_file_path=water_files[2], topol_file_path=water_files[0]
    )
    file_reader_2 = mdsuite.file_io.chemfiles_read.ChemfilesRead(
        traj_file_path=water_files[2], topol_file_path=water_files[1]
    )
    project.add_experiment(
        name="simple_water",
        timestep=0.002,
        temperature=300.0,
        units=gmx_units,
        simulation_data=file_reader_1,
        update_with_pubchempy=True,
    )
    project.add_experiment(
        name="ligand_water",
        timestep=0.002,
        temperature=300.0,
        units=gmx_units,
        simulation_data=file_reader_2,
        update_with_pubchempy=True,
    )

    project.add_experiment(
        "bmim_bf4", simulation_data=bmim_file, update_with_pubchempy=True
    )

    project.run.CoordinateUnwrapper()

    return project


class TestMoleculeMapping:
    """Class to wrap test suite so we can run all tests within PyCharm."""

    def test_water_molecule_smiles(self, mdsuite_project):
        """
        Test that water molecules are built correctly using a SMILES string. Also check
        that the molecule information is stored correctly in the experiment.

        Parameters
        ----------
        mdsuite_project : Callable
                Callable that returns an MDSuite project created in a temporary
                directory.

        Returns
        -------
        Tests that the molecule groups detected are done so correctly and that the
        constructed trajectory is also correct.

        """
        reference_molecules = {
            "water": MoleculeInfo(
                name="water",
                properties=[],
                n_particles=14,
                mass=18.015,
                groups={
                    "0": {"H": [0, 1], "O": [0]},
                    "1": {"H": [2, 3], "O": [1]},
                    "2": {"H": [4, 5], "O": [2]},
                    "3": {"H": [6, 7], "O": [3]},
                    "4": {"H": [8, 9], "O": [4]},
                    "5": {"H": [10, 11], "O": [5]},
                    "6": {"H": [12, 13], "O": [6]},
                    "7": {"H": [14, 15], "O": [7]},
                    "8": {"H": [16, 17], "O": [8]},
                    "9": {"H": [18, 19], "O": [9]},
                    "10": {"H": [20, 21], "O": [10]},
                    "11": {"H": [22, 23], "O": [11]},
                    "12": {"H": [24, 25], "O": [12]},
                    "13": {"H": [26, 27], "O": [13]},
                },
            )
        }
        water_molecule = mdsuite.Molecule(
            name="water", smiles="[H]O[H]", amount=14, cutoff=1.7, mol_pbc=True
        )
        mdsuite_project.experiments["simple_water"].run.MolecularMap(
            molecules=[water_molecule]
        )
        molecules = mdsuite_project.experiments["simple_water"].molecules
        assert molecules == reference_molecules
        assert "water" not in mdsuite_project.experiments["simple_water"].species

    def test_water_molecule_reference_dict(self, mdsuite_project):
        """
        Test that water molecules are built correctly using a reference dict.

        Parameters
        ----------
        mdsuite_project : Callable
                Callable that returns an MDSuite project created in a temporary
                directory.

        Returns
        -------
        Tests that the molecule groups detected are done so correctly and that the
        constructed trajectory is also correct.

        """
        mdsuite_project.experiments["ligand_water"].species["OW"].mass = [15.999]
        mdsuite_project.experiments["ligand_water"].species["HW1"].mass = [1.00784]
        mdsuite_project.experiments["ligand_water"].species["HW2"].mass = [1.00784]
        reference_molecules = {
            "water": MoleculeInfo(
                name="water",
                properties=[],
                n_particles=14,
                mass=18.01468,
                groups={
                    "0": {"HW1": [0], "OW": [0], "HW2": [0]},
                    "1": {"HW1": [1], "OW": [1], "HW2": [1]},
                    "2": {"HW1": [2], "OW": [2], "HW2": [2]},
                    "3": {"HW1": [3], "OW": [3], "HW2": [3]},
                    "4": {"HW1": [4], "OW": [4], "HW2": [4]},
                    "5": {"HW1": [5], "OW": [5], "HW2": [5]},
                    "6": {"HW1": [6], "OW": [6], "HW2": [6]},
                    "7": {"HW1": [7], "OW": [7], "HW2": [7]},
                    "8": {"HW1": [8], "OW": [8], "HW2": [8]},
                    "9": {"HW1": [9], "OW": [9], "HW2": [9]},
                    "10": {"HW1": [10], "OW": [10], "HW2": [10]},
                    "11": {"HW1": [11], "OW": [11], "HW2": [11]},
                    "12": {"HW1": [12], "OW": [12], "HW2": [12]},
                    "13": {"HW1": [13], "OW": [13], "HW2": [13]},
                },
            )
        }
        water_molecule = mdsuite.Molecule(
            name="water",
            species_dict={"OW": 1, "HW1": 1, "HW2": 1},
            amount=14,
            cutoff=1.7,
            mol_pbc=True,
        )
        mdsuite_project.experiments["ligand_water"].run.MolecularMap(
            molecules=[water_molecule]
        )
        molecules = mdsuite_project.experiments["ligand_water"].molecules
        assert molecules == reference_molecules
        assert "water" not in mdsuite_project.experiments["ligand_water"].species

    def test_ionic_liquid(self, mdsuite_project):
        """
        Test molecule mapping on a more complex ionic liquid.

        This test will ensure that one can pass multiple molecules to the mapper as
        well as check the effect of parsing a specific reference configuration.
        """
        bmim_molecule = mdsuite.Molecule(
            name="bmim",
            species_dict={"C": 8, "N": 2, "H": 15},
            amount=50,
            cutoff=1.9,
            reference_configuration_idx=100,
        )
        bf_molecule = mdsuite.Molecule(
            name="bf4",
            smiles="[B-](F)(F)(F)F",
            amount=50,
            cutoff=2.4,
            reference_configuration_idx=100,
        )
        mdsuite_project.experiments["bmim_bf4"].run.MolecularMap(
            molecules=[bmim_molecule, bf_molecule]
        )
