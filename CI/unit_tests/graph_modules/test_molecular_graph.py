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
Module to test the molecular graph module.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from mdsuite.graph_modules.molecular_graph import (
    _apply_system_cutoff,
    build_smiles_graph,
)


@dataclass
class SmilesTestData:
    """
    A data class to be used in the smiles string unit test.

    Attributes
    ----------
    name : str
            Name of the molecule e.g. emim
    smiles_string : str
            SMILES string for the molecule e.g. CCN1C=C[N+](+C1)C
    nodes : int
            Number of nodes that will exists in the adjacency graph built from the
            smiles string. This will correspond to the number of atoms in the molecule.
    species : dict
            A dictionary of species information for the test stating how many of each
            particle species is in the group e.g. {'C': 6, 'H': 14}

    """

    name: str
    smiles_string: str
    nodes: int
    species: dict


class MockExperiment:
    """Experiment class for a unit test."""

    database_path = Path("./")


class TestMolecularGraph:
    """Class to test the molecular graph module."""

    def test_apply_system_cutoff(self):
        """
        Test the apply_system_cutoff method.

        Returns
        -------
        Checks whether or not the cutoff has been enforced.

        """
        zeros = np.array([0, 0, 0, 0, 0])
        cutoff_data = [
            [0, 6.8, 9, 11, 0],
            [12, 0.0, 4, 3.27, 1],
            [200, 38, 0, 1, 2.11],
            [5, 5, 5.8765342, 0, 5],
            [1, 7, 3, 4.8762511, 0],
        ]
        cutoff_tensor = tf.constant(cutoff_data, dtype=tf.float32)

        # Middle range cutoff
        target = tf.constant(
            [
                [0, 0, 0, 0, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0],
            ]
        )
        mask = _apply_system_cutoff(cutoff_tensor, cutoff=5)
        np.testing.assert_array_equal(np.diagonal(mask), zeros)
        np.testing.assert_array_equal(mask, target)

        # All zeros
        target = tf.zeros((5, 5))
        mask = _apply_system_cutoff(cutoff_tensor, cutoff=0)
        np.testing.assert_array_equal(np.diagonal(mask), zeros)
        np.testing.assert_array_equal(mask, target)

        # All ones
        target = tf.ones((5, 5)) - tf.eye(5)
        mask = _apply_system_cutoff(cutoff_tensor, cutoff=200.01)
        np.testing.assert_array_equal(np.diagonal(mask), zeros)
        np.testing.assert_array_equal(mask, target)

    def test_build_smiles_graph(self):
        """
        Test the build_smiles_graph method.

        Returns
        -------
        This test checks that the SMILES graphs built by the module return the correct
        molecule information for several scenarios.

        """
        emim = SmilesTestData(
            name="emim",
            smiles_string="CCN1C=C[N+](+C1)C",
            nodes=20,
            species={"C": 6, "N": 2, "H": 12},
        )
        bmim = SmilesTestData(
            name="bmim",
            smiles_string="CCCCN1C=C[N+](+C1)C",
            nodes=26,
            species={"C": 8, "N": 2, "H": 16},
        )
        pf = SmilesTestData(
            name="pf6",
            smiles_string="F[P-](F)(F)(F)(F)F",
            nodes=7,
            species={"P": 1, "F": 6},
        )
        h2o = SmilesTestData(
            name="h2o", smiles_string="[H]O[H]", nodes=3, species={"H": 2, "O": 1}
        )
        bergenin = SmilesTestData(
            name="bergenin",
            smiles_string=(
                "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2"
            ),
            nodes=39,
            species={"C": 14, "H": 16, "O": 9},
        )
        nacl = SmilesTestData(
            name="nacl", smiles_string="[Na+].[Cl-]", nodes=2, species={"Na": 1, "Cl": 1}
        )

        data = [emim, bmim, pf, h2o, bergenin, nacl]

        for item in data:
            graph_obj, species = build_smiles_graph(item.smiles_string)
            assert graph_obj.number_of_nodes() == item.nodes
            assert species == item.species
