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

import logging
from typing import List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.graph_modules.molecular_graph import MolecularGraph
from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path
from mdsuite.utils.molecule import Molecule

log = logging.getLogger(__name__)


class MolecularMap(Transformations):
    """
    Class for mapping atoms in a database to molecules.

    Attributes
    ----------
    scale_function : dict
            A dictionary referencing the memory/time scaling function of the
            transformation.
    molecules : dict
            Molecule dictionary to use as reference. e.g.

            .. code-block::

                {'emim': {'smiles': 'CCN1C=C[N+](+C1)C', 'amount': 20}, 'PF6':
                {'smiles': 'F[P-](F)(F)(F)(F)F', 'amount': 20}}

            would be the input for the emim-PF6 ionic liquid.

    """

    def __init__(self):
        """Constructor for the MolecularMap class."""
        super().__init__()
        self.molecules = None  # parsed by the user.
        self.reference_molecules = {}
        self.adjacency_graphs = {}
        self.mapping_property = mdsuite_properties.unwrapped_positions
        self.dependency = mdsuite_properties.positions
        self.scale_function = {"quadratic": {"outer_scale_factor": 5}}

    def _prepare_database_entry(self, species: str, number_of_molecules: int) -> dict:
        """
        Call some housekeeping methods and prepare for the transformations.

        Parameters
        ----------
        species
                Name of the species to be added
        number_of_molecules : int
                Number of molecules to be added to the database.

        Returns
        -------
        data_structure : dict
                A data structure for the incoming data.

        """
        # collect machine properties and determine batch size
        path = join_path(species, self.mapping_property.name)
        dataset_structure = {
            path: (number_of_molecules, self.experiment.number_of_configurations, 3)
        }
        self.database.add_dataset(
            dataset_structure
        )  # add a new dataset to the database_path
        # data_structure = {path: {"indices": np.s_[:], "columns": [0, 1, 2]}}
        data_structure = {
            path: {
                "indices": list(range(number_of_molecules)),
                "columns": [0, 1, 2],
            }
        }

        return data_structure

    def _run_dependency_check(self):
        """
        Check that dependencies are fulfilled.

        Returns
        -------
        Calls a resolve method if dependencies are not met.

        """
        for sp_name in self.experiment.species:
            path = join_path(sp_name, self.dependency.name)
            if not self.database.check_existence(path):
                self.get_prop_through_transformation(sp_name, self.dependency)

    def _get_mass_array(self, species: list) -> list:
        """
        Return an array of atom masses for the scaling.

        Parameters
        ----------
        species : list
                List of species to be loaded.

        Returns
        -------
        mass_array : list
                A list of masses.

        """
        return [self.experiment.species[item].mass for item in species]

    def _get_type_spec(self, path_list: list) -> dict:
        """
        Compute the type spec of a list of data.

        Parameters
        ----------
        path_list : list
                List of paths for which the type species must be built.

        Returns
        -------
        type_spec : dict

        """
        type_spec = {}
        for item in path_list:
            type_spec[str.encode(item)] = tf.TensorSpec(
                shape=(None, None, 3), dtype=self.dtype
            )
        type_spec.update(
            {
                str.encode("data_size"): tf.TensorSpec(shape=(), dtype=tf.int32),
            }
        )

        return type_spec

    def _get_reduced_mass_dict(self, species: dict, molecular_mass) -> dict:
        """
        Build the reduced mass dictionary.

        This is a dictionary of reduced masses for each species in a molecule,
        i.e m_species / m_molecule. These are used in the COM positions computation
        later.

        Parameters
        ----------
        species : list
                List of species to include in the dict.

        Returns
        -------
        reduced_mass_dict : dict
                Dictionary of reduced masses for each species.

        """
        reduced_mass_dict = {}
        for item in species:
            reduced_mass_dict[item] = (
                self.experiment.species[item].mass[0] / molecular_mass
            )

        return reduced_mass_dict

    def _map_molecules(self, molecular_graph: MolecularGraph):
        """
        Map the molecules and save the data in the database.

        Returns
        -------
        Updates the database.

        """
        molecule_name = molecular_graph.molecule_name
        molecules = self.experiment.molecules
        molecules[molecule_name] = {}
        molecules[molecule_name]["n_particles"] = molecular_graph.n_molecules

        molecules[molecule_name]["mass"] = molecular_graph.molecular_mass
        molecules[molecule_name]["groups"] = molecular_graph.molecular_groups
        scaling_factor = molecular_graph.molecular_mass

        mass_dictionary = self._get_reduced_mass_dict(
            molecular_graph.species, scaling_factor
        )

        # Prepare the data structures and monitors
        data_structure = self._prepare_database_entry(
            molecule_name, molecular_graph.n_molecules
        )
        path_list = [
            join_path(s, self.mapping_property.name) for s in molecular_graph.species
        ]
        self._prepare_monitors(data_path=path_list)

        type_spec = self._get_type_spec(path_list)
        batch_generator, batch_generator_args = self.data_manager.batch_generator()

        data_set = tf.data.Dataset.from_generator(
            batch_generator, args=batch_generator_args, output_signature=type_spec
        )
        data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)

        log.info(f"Mapping molecule graphs onto trajectory for {molecule_name}")
        for i, batch in tqdm(enumerate(data_set), ncols=70, total=self.n_batches):
            batch_size = batch[b"data_size"]

            trajectory = np.zeros(
                shape=(
                    molecular_graph.n_molecules,
                    batch_size,
                    3,
                )
            )

            for t, molecule in enumerate(molecular_graph.molecular_groups):
                # Load species molecule-specific particles into a separate dict
                # and apply their respective scaling factor.
                molecule_trajectory = np.zeros((batch_size, 3))
                for item in molecular_graph.molecular_groups[molecule]:
                    batch_reference = str.encode(f"{item}/{self.mapping_property.name}")
                    particles = molecular_graph.molecular_groups[molecule][item]
                    particle_trajectories = (
                        tf.gather(batch[batch_reference], particles)
                        * mass_dictionary[item]
                    )
                    molecule_trajectory += tf.reduce_sum(particle_trajectories, axis=0)

                # Compute the COM trajectory
                trajectory[t] = molecule_trajectory

            self._save_output(
                data=trajectory,
                data_structure=data_structure,
                index=i * self.batch_size,
            )

            self.experiment.molecules = molecules

    def run_transformation(self, molecules: List[Molecule]):
        """
        Perform the transformation.

        Parameters
        ----------
        molecules : List[Molecule]
                A list of MDSuite Molecule objects. For each, a molecule will be
                mapped.

        Returns
        -------
        Update the experiment database.

        """
        self._run_dependency_check()

        # Populate the molecules dict
        for item in molecules:
            molecular_graph = MolecularGraph(
                experiment=self.experiment,
                molecule_input_data=item,
            )
            if item.mol_pbc:
                self.mapping_property = mdsuite_properties.positions

            self._map_molecules(molecular_graph)

            if item.mol_pbc:
                self.experiment.run.CoordinateUnwrapper(species=[item.name])
            else:
                self.experiment.run.CoordinateWrapper(species=[item.name])
