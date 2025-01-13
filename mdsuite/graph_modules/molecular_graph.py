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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from pysmiles import read_smiles
from tqdm import tqdm

from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.database.simulation_database import Database
from mdsuite.utils.meta_functions import check_a_in_b, join_path
from mdsuite.utils.molecule import Molecule

log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from mdsuite.experiment import Experiment


class MolecularGraph:
    """
    Class for building and studying molecular graphs.

    Attributes
    ----------
    reference_property : str
            MDSuite property to use for reference during the unwrapping.

    """

    molecular_mass: float
    molecular_groups: dict

    def __init__(
        self,
        experiment: Experiment,
        molecule_input_data: Molecule,
    ):
        """
        Constructor for the MolecularGraph class.

        Parameters
        ----------
        experiment : Experiment
                Experiment object from which to read.
        molecule_input_data :An MDSuite Molecule instance.

        """
        self.experiment = experiment
        self.molecule_name = molecule_input_data.name
        self.database = Database(self.experiment.database_path / "database.hdf5")
        self.cutoff = molecule_input_data.cutoff
        self.n_molecules = molecule_input_data.amount
        self.mol_pbc = molecule_input_data.mol_pbc

        if self.mol_pbc:
            self.reference_property = mdsuite_properties.positions
        else:
            self.reference_property = mdsuite_properties.unwrapped_positions

        if isinstance(molecule_input_data.reference_configuration_idx, int):
            self.reference_configuration = molecule_input_data.reference_configuration_idx
        else:
            self.reference_configuration = 0

        if isinstance(molecule_input_data.smiles, str):
            self.smiles_graph, self.species = build_smiles_graph(
                molecule_input_data.smiles
            )
        elif isinstance(molecule_input_data.species_dict, dict):
            self.species = molecule_input_data.species_dict
            self.smiles_string = None
        else:
            error_msg = (
                "The minimum amount of data was not given to the mapping."
                "Either provide a reference key with information about"
                "Which species and the number of them are in the molecule,"
                "or provide a SMILES string that can be used to compute "
                "this information."
            )
            raise ValueError(error_msg)

        self._get_molecular_mass()
        self._build_molecule_groups()  # populate the class group attribute
        self._perform_isomorphism_tests()  # run the graph tests.

    def _get_molecular_mass(self):
        """
        Get the mass of a SMILES molecule based on experiment data.

        Returns
        -------
        Updates the following class attributes:

        mass : float
                mass of the molecule

        """
        self.molecular_mass = 0.0
        for species, number in self.species.items():
            self.molecular_mass += self.experiment.species[species].mass[0] * number
        # we round to .14f to avoid some very small uncertainties and test equality
        self.molecular_mass = round(self.molecular_mass, 14)

    def build_configuration_graph(self) -> tf.Tensor:
        """
        Build a graph for the configuration.

        Returns
        -------
        adjacency_matrix : tf.Tensor
                An adjacency matrix for the configuration describing which atoms are
                bonded to which others.

        """
        path_list = [
            join_path(species, self.reference_property.name) for species in self.species
        ]
        data_dict = self.database.load_data(
            path_list=path_list, select_slice=np.s_[:, self.reference_configuration]
        )
        data = []
        for item in path_list:
            data.append(data_dict[item])
        configuration_tensor = tf.concat(data, axis=0)
        distance_matrix = get_neighbour_list(
            configuration_tensor, cell=self.experiment.box_array
        )

        return _apply_system_cutoff(distance_matrix, self.cutoff)

    def _build_molecule_groups(self):
        """
        Build molecule groups from decomposed graph.

        Returns
        -------

        """
        adjacency_graph = self.build_configuration_graph()
        decomposed_graphs = self._perform_graph_decomposition(adjacency_graph)
        self.molecular_groups = self._split_decomposed_graphs(decomposed_graphs)

    def _perform_graph_decomposition(self, adjacency_matrix: tf.Tensor) -> dict:
        """
        Reduce an adjacency matrix into a linear combination of sub-matrices.

        This is the process of graph decomposition in which one large graph is
        decomposed into smaller, independent graphs. In the case of this data, these
        sub-graphs are for a single molecule, therefore, there should be one sub-graph
        per molecule for each species.

        Parameters
        ----------
        adjacency_matrix : tf.Tensor
                Adjacency tensor to reduce.

        Returns
        -------
        reduced_graphs : dict
                A dict of sub graphs constructed from the decomposition of the adjacency
                matrix. Of the form {'0': [], '1': []}

        """
        # TODO: wrap this in an optimizer to iteratively improve the cutoff until the
        #       number is correct.

        molecules = {}
        log.info(f"Building molecular graph from configuration for {self.molecule_name}")
        # TODO speed up
        for i in tqdm(range(len(adjacency_matrix)), ncols=70):
            indices = tf.where(adjacency_matrix[i])
            indices = tf.reshape(indices, -1)
            if len(molecules) == 0:
                molecule = 0
                molecules[molecule] = indices
            else:
                molecule = None
                for mol in molecules:
                    if check_a_in_b(indices, molecules[mol]):
                        molecule = mol
                        molecules[mol] = tf.concat([molecules[mol], indices], 0)
                        molecules[mol] = tf.unique(molecules[mol])[0]
                        break
                if molecule is None:
                    molecule = len(molecules)
                    molecules[molecule] = indices

        del_list = []
        for item in molecules:
            test_dict = molecules.copy()
            test_dict.pop(item)
            for reference in test_dict:
                if all(elem in test_dict[reference] for elem in molecules[item]):
                    del_list.append(item)

        for item in del_list:
            molecules.pop(item)

        return molecules

    def _perform_isomorphism_tests(self):
        """
        Run isomorphism checks to determine whether or not the graphs computed are
        correct.

        Currently runs the following tests:

        1. Checks that the number of decomposed graphs is equal to the number of
           expected molecules.
        2. Checks that the number of particles of each constituent species for each
           molecule matches that given by the SMILES string or the user provided
           reference data.
        """
        # amount of molecules test
        self._amount_isomorphism_test()
        # groups equality test
        self._molecule_group_equality_isomorphism_test()

    def _amount_isomorphism_test(self):
        """
        Test that the amount of computed molecules is equal to the expected amount.

        Returns
        -------
        Returns nothing, raises a value error if condition is not met.

        """
        log.info("Performing molecule number isomorphism test.")
        # number of molecules test
        if self.n_molecules is None:
            log.info("No molecule amount to check against, skipping test.")
        else:
            if len(self.molecular_groups) != self.n_molecules:
                raise ValueError(
                    f"Expected number of molecules ({self.n_molecules}) does not "
                    f"match the amount computed ({len(self.molecular_groups)}), "
                    "please adjust cutoff parameters."
                )
            else:
                log.info("Amount of molecules test passed.")

    def _molecule_group_equality_isomorphism_test(self):
        """
        Test that the molecule groups computed match that of the reference.

        Returns
        -------
        Nothing, will raise an exception if the test fails.

        """
        log.info("Performing group equality isomorphism test.")
        for mol_number, mol_data in self.molecular_groups.items():
            for species, indices in mol_data.items():
                if not len(indices) == self.species[species]:
                    error_msg = (
                        f"Molecule group {mol_number}, with molecule data {mol_data},"
                        f"did not match with the reference data in {self.species}."
                    )
                    raise AssertionError(error_msg)

        log.info("Group equality isomorphism test passed.")

    def _adjacency_graph_isomorphism_test(self):
        """
        Determine approximate isomorphism between the computed adjacency graph and a
        reference graph.

        Returns
        -------
        Nothing, will raise an exception if the test fails.

        Notes
        -----
        This must be implemented, however, will be quite an expensive operation.

        """
        raise NotImplementedError

    def _split_decomposed_graphs(self, graph_dict: dict) -> dict:
        """
        Build an indices dict to store the groups of atoms in each molecule.

        Parameters
        ----------
        graph_dict : dict
                Dict of decomposed graphs to be converted into correct particle species
                indices.

        Returns
        -------
        group_dict : dict
                A dictionary of atoms and indices that specify that indices of
                this species is in a molecule.

        """
        particle_groups = {}
        for item in graph_dict:
            indices_dict = {}
            lengths = [self.experiment.species[item].n_particles for item in self.species]
            lengths = np.cumsum(lengths)

            for i, particle_species in enumerate(self.species):
                if i == 0:
                    indices_dict[particle_species] = np.sort(
                        np.array(list(filter(lambda x: x < lengths[i], graph_dict[item])))
                    ).tolist()
                else:
                    greater_array = list(
                        filter(lambda x: x >= lengths[i - 1], graph_dict[item])
                    )
                    constrained_array = list(
                        filter(lambda x: x < lengths[i], greater_array)
                    )
                    indices_dict[particle_species] = np.sort(
                        np.array(constrained_array) - (lengths[i - 1])
                    ).tolist()

            particle_groups[item] = indices_dict

        return particle_groups


def build_smiles_graph(smiles_string: str) -> tuple:
    """
    Build molecular graphs from SMILES strings.

    Parameters
    ----------
    smiles_string : str
            SMILES string to use in the graph construction.

    Returns
    -------
    smiles_graph :
            Graph object returned by PySmiles
    species : dict
            A dict object containing species information about the molecule.

    """
    mol = read_smiles(smiles_string, explicit_hydrogen=True)
    data = mol.nodes
    species = {}
    for i in range(len(data)):
        item = data[i].get("element")
        if item in species:
            species[item] += 1
        else:
            species[item] = 1

    return mol, species


def _apply_system_cutoff(input_tensor: tf.Tensor, cutoff: float) -> tf.Tensor:
    """
    Enforce a cutoff on a tensor.

    In this context the cutoff is used to identify bonded atoms. We argue
    that the closest atoms will be bonded as defined by a cutoff.
    Constructing the mask of closest atoms will allow for the bonded ones to be
    identified.

    Parameters
    ----------
    input_tensor : tf.Tensor
            Tensor of any size or shape to be masked. In our case it is a distance
            tensor of the atoms in a configuration.
    cutoff : float
            Cutoff to use in the mask. If a distance is greater than this cutoff it
            is marked as 0, if not, it is 1.

    Returns
    -------
    masked_tensor : tf.Tensor
            A tensor of ones and zeros where 1s corresponded to 'bonded' particles
            and 0s indicated no bonding. Note, the diagonals of this tensor are
            set to 0 as a particle cannot bond itself.

    """
    cutoff_mask = tf.cast(
        tf.less(input_tensor, cutoff), dtype=tf.int16
    )  # Construct the mask

    return tf.linalg.set_diag(cutoff_mask, np.zeros(len(input_tensor)))


def get_neighbour_list(positions: tf.Tensor, cell: list = None) -> tf.Tensor:
    """
    Generate the neighbour list.

    Parameters
    ----------
    positions: tf.Tensor
        Tensor with shape (number_of_configurations, n_atoms, 3)
        representing the coordinates
    cell: list
        If periodic boundary conditions are used, please supply the cell
        dimensions, e.g. [13.97, 13.97, 13.97]. If the cell is provided
        minimum image convention will be applied!

    Returns
    -------
    neighbour_list : tf.Tensor
            Neighbour list for a single configuration.

    """
    r_ij_matrix = tf.reshape(positions, (1, len(positions), 3)) - tf.reshape(
        positions, (len(positions), 1, 3)
    )

    # Pretty sure we never need min image for mapping.
    if cell:
        r_ij_matrix -= tf.math.rint(r_ij_matrix / cell) * cell
    return tf.norm(r_ij_matrix, ord="euclidean", axis=2)
