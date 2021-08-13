"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Parent class for MDSuite transformations
"""
import time
from typing import Union
import os
import numpy as np
import tensorflow as tf
from mdsuite.memory_management.memory_manager import MemoryManager
from mdsuite.database.data_manager import DataManager
from mdsuite.database.simulation_database import Database
from mdsuite.utils.meta_functions import join_path

switcher_transformations = {
    "Translational_Dipole_Moment": "TranslationalDipoleMoment",
    "Ionic_Current": "IonicCurrent",
    "Integrated_Heat_Current": "IntegratedHeatCurrent",
    "Thermal_Flux": "ThermalFlux",
    "Momentum_Flux": "MomentumFlux",
    "Kinaci_Heat_Current": "KinaciIntegratedHeatCurrent",
}


class Transformations:
    """
    Parent class for MDSuite transformations.

    Attributes
    ----------
    database : Database
            database class object for data loading and storing
    experiment : object
            Experiment class instance to update
    batch_size : int
            batch size for the computation
    n_batches : int
            Number of batches to be looped over
    remainder : int
            Remainder amount to add after the batches are looped over.
    data_manager : DataManager
            data manager for handling the data transfer
    memory_manager : MemoryManager
            memory manager for the computation.
    """

    def __init__(self, experiment: object):
        """
        Constructor for the experiment class

        Parameters
        ----------
        experiment : object
                Experiment class object to update
        """
        self.experiment = experiment
        self.database = Database(
            name=os.path.join(self.experiment.database_path, "database.hdf5"),
            architecture="simulation",
        )
        self.batch_size: int
        self.n_batches: int
        self.remainder: int

        self.dependency = None
        self.scale_function = None
        self.offset = 0

        self.data_manager: DataManager
        self.memory_manager: MemoryManager

    def _run_dataset_check(self, path: str):
        """
        Check to see if the database dataset already exists. If it does, the transformation should extend the dataset
        and add data to the end of it rather than try to add data.

        Parameters
        ----------
        path : str
                dataset path to check.
        Returns
        -------
        outcome : bool
                If True, the dataset already exists and should be extended. If False, a new dataset should be built.
        """
        return self.database.check_existence(path)

    def _run_dependency_check(self):
        """
        Check that dependencies are fulfilled.

        Returns
        -------
        Calls a resolve method if dependencies are not met.
        """
        path_list = []
        truth_array = [
            join_path(species, self.dependency) for species in self.experiment.species
        ]
        for item in path_list:
            truth_array.append(self.database.check_existence(item))
        if all(truth_array):
            return
        else:
            self._resolve_dependencies(self.dependency)

    def _resolve_dependencies(self, dependency):
        """
        Resolve any calculation dependencies if possible.

        Parameters
        ----------
        dependency : str
                Name of the dependency to resolve.

        Returns
        -------

        """

        def _string_to_function(argument):
            """
            Select a transformation based on an input

            Parameters
            ----------
            argument : str
                    Name of the transformation required

            Returns
            -------
            transformation call.
            """

            switcher_unwrapping = {
                "Unwrapped_Positions": self._unwrap_choice(),
            }

            switcher = {**switcher_unwrapping, **switcher_transformations}

            choice = switcher.get(
                argument, lambda: "Data not in database and can not be generated."
            )
            return choice

        transformation = _string_to_function(dependency)
        self.experiment.perform_transformation(transformation)

    def _unwrap_choice(self):
        """
        Unwrap either with indices or with box arrays.
        Returns
        -------

        """
        indices = self.database.check_existence("Box_Images")
        if indices:
            return "UnwrapViaIndices"
        else:
            return "UnwrapCoordinates"

    def _update_type_dict(self, dictionary: dict, path_list: list, dimension: int):
        """
        Update a type spec dictionary.

        Parameters
        ----------
        dictionary : dict
                Dictionary to append
        path_list : list
                List of paths for the dictionary
        dimension : int
                Dimension of the property
        Returns
        -------
        type dict : dict
                Dictionary for the type spec.
        """
        for item in path_list:
            dictionary[str.encode(item)] = tf.TensorSpec(
                shape=(None, None, dimension), dtype=tf.float64
            )

        return dictionary

    def _update_species_type_dict(
        self, dictionary: dict, path_list: list, dimension: int
    ):
        """
        Update a type spec dictionary for a species input.

        Parameters
        ----------
        dictionary : dict
                Dictionary to append
        path_list : list
                List of paths for the dictionary
        dimension : int
                Dimension of the property
        Returns
        -------
        type dict : dict
                Dictionary for the type spec.
        """
        for item in path_list:
            species = item.split("/")[0]
            n_atoms = len(self.experiment.species[species]["indices"])
            dictionary[str.encode(item)] = tf.TensorSpec(
                shape=(n_atoms, None, dimension), dtype=tf.float64
            )

        return dictionary

    def _remainder_to_binary(self):
        """
        If a remainder is > 0, return 1, else, return 0
        Returns
        -------
        binary_map : int
                If remainder > 0, return 1, else,  return 0
        """
        return int(self.remainder > 0)

    def _save_coordinates(
        self,
        data: Union[tf.Tensor, np.array],
        index: int,
        batch_size: int,
        data_structure: dict,
        system_tensor: bool = True,
        tensor: bool = False,
    ):
        """
        Save the tensor_values into the database_path

        Returns
        -------
        saves the tensor_values to the database_path.
        """
        try:
            self.database.add_data(
                data=data,
                structure=data_structure,
                start_index=index + self.offset,
                batch_size=batch_size,
                system_tensor=system_tensor,
                tensor=tensor,
            )
        except OSError:
            # This is used because in Windows and in WSL we got the error that the file
            # was still open while it should already be closed. So, we wait, and we add again.
            time.sleep(0.5)
            self.database.add_data(
                data=data,
                structure=data_structure,
                start_index=index + self.offset,
                batch_size=batch_size,
                system_tensor=system_tensor,
                tensor=tensor,
            )

    def _prepare_monitors(self, data_path: Union[list, np.array]):
        """
        Prepare the tensor_values and memory managers.

        Parameters
        ----------
        data_path : list
                List of tensor_values paths to load from the hdf5 database_path.

        Returns
        -------

        """
        self.memory_manager = MemoryManager(
            data_path=data_path,
            database=self.database,
            memory_fraction=0.5,
            scale_function=self.scale_function,
            offset=self.offset,
        )
        (
            self.batch_size,
            self.n_batches,
            self.remainder,
        ) = self.memory_manager.get_batch_size()
        self.data_manager = DataManager(
            data_path=data_path,
            database=self.database,
            batch_size=self.batch_size,
            n_batches=self.n_batches,
            remainder=self.remainder,
            offset=self.offset,
        )

    def run_transformation(self):
        """
        Perform the transformation
        """
        raise NotImplementedError  # implemented in child class.
