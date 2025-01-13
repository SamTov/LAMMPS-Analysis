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
import logging
import pathlib
import time
import typing
from typing import List

import h5py as hf
import numpy as np
import tensorflow as tf

from mdsuite.utils.meta_functions import join_path

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PropertyInfo:
    """
    Information of a trajectory property.

    Example:
    -------
    pos_info = PropertyInfo('Positions', 3)
    vel_info = PropertyInfo('Velocities', 3).

    Attributes:
    ----------
    name:
        The name of the property
    n_dims:
        The dimensionality of the property

    """

    name: str
    n_dims: int


@dataclasses.dataclass
class SpeciesInfo:
    """
    Information of a species.

    Attributes
    ----------
    name
        Name of the species (e.g. 'Na')
    n_particles
        Number of particles of that species
    properties: list of PropertyInfo
        List of the properties that were recorded for the species
        mass and charge are optional

    """

    name: str
    n_particles: int
    properties: List[PropertyInfo]
    mass: float = None
    charge: float = 0

    def __eq__(self, other):
        same = (
            self.name == other.name
            and self.n_particles == other.n_particles
            and self.mass == other.mass
            and self.charge == other.charge
        )
        if len(self.properties) != len(other.properties):
            return False

        for prop_s, prop_o in zip(self.properties, other.properties):
            same = same and prop_s == prop_o
        return same


@dataclasses.dataclass
class MoleculeInfo(SpeciesInfo):
    """Information about a Molecule.

    All the information of a species + groups

    Attributes
    ----------
    groups: dict
        A molecule specific dictionary for mapping the molecule to the
        particles. The keys of this dict are index references to a specific molecule,
        i.e. molecule 1 and the values are a dict of atom species and their indices
        belonging to that specific molecule.
        e.g
            water = {"groups": {"0": {"H": [0, 1], "O": [0]}}
        This tells us that the 0th water molecule consists of the 0th and 1st hydrogen
        atoms in the database as well as the 0th oxygen atom.

    """

    groups: dict = None

    def __eq__(self, other):
        """Add a check to see if the groups are identical."""
        if self.groups != other.groups:
            return False
        return super(MoleculeInfo, self).__eq__(other)


@dataclasses.dataclass
class TrajectoryMetadata:
    """Trajectory Metadata container.

    This metadata must be extracted from trajectory files to build the database into
    which the trajectory will be stored.

    Attributes
    ----------
    n_configurations : int
        Number of configurations of the whole trajectory.
    species_list: list of SpeciesInfo
        The information about all species in the system.
    box_l: list of float
        The simulation box size in three dimensions
    sample_rate : int optional
        The number of timesteps between consecutive samples
        # todo remove in favour of sample_step
    sample_step : int optional
        The time between consecutive configurations.
        E.g. for a simulation with time step 0.1 where the trajectory is written
        every 5 steps: sample_step = 0.5. Does not have to be specified
        (e.g. configurations from Monte Carlo scheme), but is needed for all
        dynamic observables.
    temperature : float optional
        The set temperature of the system.
        Optional because only applicable for MD simulations with thermostat.
        Needed for certain observables.
    simulation_data : str|Path, optional
        All other simulation data that can be extracted from the trajectory metadata.
        E.g. software version, pressure in NPT simulations, time step, ...

    """

    n_configurations: int
    species_list: List[SpeciesInfo]
    box_l: list = None
    sample_rate: int = 1
    sample_step: float = None
    temperature: float = None
    simulation_data: dict = dataclasses.field(default_factory=dict)


class TrajectoryChunkData:
    """Class to specify the data format for transfer from the file to the database."""

    def __init__(self, species_list: List[SpeciesInfo], chunk_size: int):
        """

        Parameters
        ----------
        species_list : List[SpeciesInfo]
            List of SpeciesInfo.
            It contains the information which species are there and which properties
            are recorded for each
        chunk_size : int
            The number of configurations to be stored in this chunk

        """
        self.chunk_size = chunk_size
        self.species_list = species_list
        self._data = {}
        for sp_info in species_list:
            self._data[sp_info.name] = {}
            for prop_info in sp_info.properties:
                self._data[sp_info.name][prop_info.name] = np.zeros(
                    (chunk_size, sp_info.n_particles, prop_info.n_dims)
                )

    def add_data(self, data: np.ndarray, config_idx, species_name, property_name):
        """
        Add configuration data to the chunk

        Parameters
        ----------
        data:
            The data to be added, with shape (n_configs, n_particles, n_dims).
            n_particles and n_dims relates to the species and the property that is
            being added
        config_idx:
            Start index of the configs that are being added.
        species_name
            Name of the species to which the data belongs
        property_name
            Name of the property being added.

        Example:
        -------
        Storing velocity Information for 42 Na atoms in the 17th iteration of a loop
        that reads 5 configs per loop:
        add_data(vel_array, 16*5, 'Na', 'Velocities')
        where vel.data.shape == (5,42,3)

        """
        n_configs = len(data)
        self._data[species_name][property_name][
            config_idx : config_idx + n_configs, :, :
        ] = data

    def get_data(self):
        return self._data


class Database:
    """
    Database class.

    Databases make up a large part of the functionality of MDSuite and are kept
    fairly consistent in structure. Therefore, the database_path structure we
    are using has a separate class with commonly used methods which act as
    wrappers for the hdf5 database_path.

    Attributes
    ----------
    path : str|Path
            The name of the database_path in question.

    """

    def __init__(self, path: typing.Union[str, pathlib.Path] = "database"):
        """
        Constructor for the database_path class.

        Parameters
        ----------
        path : str|Path
                The name of the database_path in question.

        """
        if isinstance(path, pathlib.Path):
            self.path = path.as_posix()
        elif isinstance(path, str):
            self.path = path  # name of the database_path
        else:
            # TODO fix this!
            log.debug(f"Expected str|Path but found {type(path)}")
            self.path = path

    @staticmethod
    def _update_indices(
        data: np.array, reference: np.array, batch_size: int, n_atoms: int
    ):
        """
        Update the indices key of the structure dictionary if the tensor_values must be
        sorted.

        Parameters
        ----------
        data : np.ndarray
        reference : np.ndarray
        batch_size : int
        n_atoms : int

        Returns
        -------

        """
        ids = np.reshape(np.array(data[:, 0]).astype(int), (-1, n_atoms))
        ref_ids = np.argsort(ids, axis=1)
        n_batches = ids.shape[0]

        return (
            ref_ids[:, reference - 1] + (np.arange(n_batches) * n_atoms)[None].T
        ).flatten()

    @staticmethod
    def _build_path_input(structure: dict) -> dict:
        """
        Build an input to a hdf5 database_path from a dictionary.

        In many cases, whilst a dict can be passed on to a method, it is not ideal for
        use in the hdf5 database_path. This method takes a dictionary and return a new
        dictionary with the relevant file path.


        Parameters
        ----------
        structure : dict
                General structure of the dictionary with relevant dataset sizes. e.g.
                {'Na': {'Forces': (200, 5000, 3)},
                'Pressure': (5000, 6), 'Temperature': (5000, 1)} In this case, the last
                 value in the tuple corresponds to the number of components that wil be
                 parsed to the database_path. The value in the dict can also be an
                 integer corresponding to a resize operation such as
                 {'Na': {'velocities' 100}}. In any case, the deepest portion of the
                 dict must be a non-dict object and will be returned as the value of the
                 path to it in the new dictionary.


        Returns
        -------
        architecture : dict
                Corrected path in the hdf5 database_path. e.g. {'/Na/Velocities': 100},
                or {'/Na/Forces': (200, 5000, 3)}

        """
        # Build file paths for the addition.
        architecture = {}
        for group in structure:
            if type(structure[group]) is not dict:
                architecture[group] = structure[group]
            else:
                for subgroup in structure[group]:
                    db_path = join_path(group, subgroup)
                    architecture[db_path] = structure[group][subgroup]

        return architecture

    def add_data(self, chunk: TrajectoryChunkData):
        """
        Add new data to the dataset.

        Parameters
        ----------
        chunk:
            a data chunk
        start_idx:
            Configuration at which to start writing.

        """
        workaround_time_in_axis_1 = True

        chunk_data = chunk.get_data()

        with hf.File(self.path, "r+") as database:
            for sp_info in chunk.species_list:
                for prop_info in sp_info.properties:
                    dataset_name = f"{sp_info.name}/{prop_info.name}"
                    write_data = chunk_data[sp_info.name][prop_info.name]

                    dataset_shape = database[dataset_name].shape
                    start_index = database[dataset_name].attrs["starting_index"]
                    stop_index = start_index + chunk.chunk_size

                    if len(dataset_shape) == 2:
                        # only one particle
                        database[dataset_name][start_index:stop_index, :] = write_data[
                            :, 0, :
                        ]

                    elif len(dataset_shape) == 3:
                        if workaround_time_in_axis_1:
                            database[dataset_name][:, start_index:stop_index, :] = (
                                np.swapaxes(write_data, 0, 1)
                            )
                        else:
                            database[dataset_name][
                                start_index:stop_index, ...
                            ] = write_data
                    else:
                        raise ValueError(
                            "dataset shape must be either (n_part,n_config,n_dim) or"
                            " (n_config, n_dim)"
                        )
                    database[dataset_name].attrs["starting_index"] += chunk.chunk_size

    def resize_datasets(self, structure: dict):
        """
        Resize a dataset so more tensor_values can be added.

        Parameters
        ----------
        structure : dict
                path to the dataset that needs to be resized. e.g.
                {'Na': {'velocities': (32, 100, 3)}}
                will resize all 'x', 'y', and 'z' datasets by 100 entries.

        Returns
        -------

        """
        with hf.File(self.path, "r+") as db:
            # construct the architecture dict
            architecture = self._build_path_input(structure=structure)

            # Check for a type error in the dataset information
            for identifier in architecture:
                dataset_information = architecture[identifier]
                if not isinstance(dataset_information, tuple):
                    raise TypeError("Invalid input for dataset generation")

                # get the correct maximum shape for the dataset -- changes if an
                # experiment property or an atomic property
                try:
                    if len(dataset_information[:-1]) == 1:
                        axis = 0
                    else:
                        axis = 1

                    expansion = dataset_information[axis] + db[identifier].shape[axis]
                    db[identifier].resize(expansion, axis)

                # It is actually a new group
                except KeyError:
                    self.add_dataset({identifier: architecture[identifier]})

    def initialize_database(self, structure: dict):
        """
        Build a database_path with a general structure.

        Note, this method WILL overwrite a pre-existing database_path. This is because
        it is only to be called on the initial construction of an experiment class and
        the first addition of tensor_values to it.


        Parameters
        ----------
        structure : dict
                General structure of the dictionary with relevant dataset sizes.
                e.g. {'Na': {'Forces': (200, 5000, 3)}, 'Pressure': (5000, 6),
                'Temperature': (5000, 1)} In this case, the last value in the tuple
                corresponds to the number of components that wil be parsed to the
                database_path.

        Returns
        -------

        """
        architecture = self._build_path_input(structure)
        self.add_dataset(architecture)  # add a dataset to the groups

    def database_exists(self) -> bool:
        """Check if the database file already exists."""
        return pathlib.Path(self.path).exists()

    def add_dataset(self, architecture: dict):
        """
        Add a dataset of the necessary size to the database_path.

        Just as a separate method exists for building the group structure of the hdf5
        database_path, so too do we include a separate method for adding a dataset.
        This is so datasets can be added not just upon the initial construction of the
        database_path, but also if tensor_values is added in the future that should
        also be stored. This method will assume that a group has already been built,
        although this is not necessary for HDF5, the separation of the actions is good
        practice.

        Parameters
        ----------
        architecture : dict
                Structure of properties to be added to the database_path.
                e.g. {'Na': {'Forces': (200, 5000, 3)}}

        Returns
        -------
        Updates the database_path directly.

        """
        with hf.File(self.path, "a") as database:
            for item in architecture:
                dataset_information = architecture[item]  # get the tuple information
                dataset_path = item  # get the dataset path in the database_path

                # Check for a type error in the dataset information
                try:
                    if type(dataset_information) is not tuple:
                        raise TypeError("Invalid input for dataset generation")
                except TypeError:
                    raise TypeError

                if len(dataset_information[:-1]) == 1:
                    vector_length = dataset_information[-1]
                    max_shape = (None, vector_length)
                else:
                    max_shape = list(dataset_information)
                    max_shape[1] = None
                    max_shape = tuple(max_shape)

                database.create_dataset(
                    dataset_path,
                    dataset_information,
                    maxshape=max_shape,
                    compression="gzip",
                    chunks=True,
                )
                dataset = database[dataset_path]
                dataset.attrs["starting_index"] = 0

    def _add_group_structure(self, structure: dict):
        """
        Add a simple group structure to a database_path.
        This method will take an input structure and build the required group structure
        in the hdf5 database_path. This will NOT however instantiate the dataset for the
        structure, only the group hierarchy.


        Parameters
        ----------
        structure : dict
                Structure of a single property to be added to the database_path.
                e.g. {'Na': {'Forces': (200, 5000, 3)}}

        Returns
        -------
        Updates the database_path directly.

        """
        with hf.File(self.path, "a") as database:
            # Build file paths for the addition.
            architecture = self._build_path_input(structure=structure)
            for item in list(architecture):
                if item in database:
                    log.info("Group structure already exists")
                else:
                    database.create_group(item)

    def get_memory_information(self) -> dict:
        """
        Get memory information from the database_path.

        Returns
        -------
        memory_database : dict
                A dictionary of the memory information of the groups in the
                database_path

        """
        with hf.File(self.path, "r") as database:
            memory_database = {}
            for item in database:
                for ds in database[item]:
                    memory_database[join_path(item, ds)] = database[item][ds].nbytes

        return memory_database

    def check_existence(self, path: str) -> bool:
        """
        Check to see if a dataset is in the database_path.

        Parameters
        ----------
        path : str
                Path to the desired dataset

        Returns
        -------
        response : bool
                If true, the path exists, else, it does not.

        """
        with hf.File(self.path, "r") as database_object:
            keys = []
            database_object.visit(
                lambda item: (
                    keys.append(database_object[item].name)
                    if isinstance(database_object[item], hf.Dataset)
                    else None
                )
            )
            path = f"/{path}"  # add the / to avoid name overlapping

            response = any(list(item.endswith(path) for item in keys))
        return response

    def change_key_names(self, mapping: dict):
        """
        Change the name of database_path keys.

        Parameters
        ----------
        mapping : dict
                Mapping for the change of names

        Returns
        -------
        Updates the database_path

        """
        with hf.File(self.path, "r+") as db:
            groups = list(db.keys())

            for item in groups:
                if item in mapping:
                    db.move(item, mapping[item])

    def load_data(
        self,
        path_list: list = None,
        select_slice: np.s_ = np.s_[:],
        dictionary: bool = False,
        scaling: list = None,
        d_size: int = None,
    ):
        """
        Load tensor_values from the database_path for some operation.

        Should be called by the tensor_values fetch class as this will ensure
        correct loading and pre-loading.

        Returns
        -------

        """
        if scaling is None:
            scaling = [1 for _ in range(len(path_list))]

        with hf.File(self.path, "r") as database:
            data = {}
            for i, item in enumerate(path_list):
                if type(select_slice) is dict:
                    # index is the particle species name in the full item as a str.
                    slice_index = item.decode().split("/")[0]
                    my_slice = select_slice[slice_index]
                else:
                    my_slice = select_slice
                try:
                    data[item] = (
                        tf.convert_to_tensor(database[item][my_slice], dtype=tf.float64)
                        * scaling[i]
                    )
                except TypeError:
                    data[item] = (
                        tf.convert_to_tensor(
                            database[item][my_slice[0]][:, my_slice[1], :],
                            dtype=tf.float64,
                        )
                        * scaling[i]
                    )
            data[str.encode("data_size")] = d_size

        return data

    def get_load_time(self, database_path: str = None):
        """
        Calculate the open/close time of the database_path.

        Parameters
        ----------
        database_path : str
                Database path on which to test the time.

        Returns
        -------
        opening time : float
                Time taken to open and close the database_path

        """
        if database_path is None:
            start = time.time()
            database_path = hf.File(self.path, "r")
            database_path.close()
            stop = time.time()
        else:
            start = time.time()
            database_path = hf.File(database_path, "r")
            database_path.close()
            stop = time.time()

        return stop - start

    def get_data_size(self, data_path: str) -> tuple:
        """
        Return the size of a dataset as a tuple (n_rows, n_columns, n_bytes).

        Parameters
        ----------
        data_path : str
                path to the tensor_values in the hdf5 database_path.

        Returns
        -------
        dataset_properties : tuple
                Tuple of tensor_values about the dataset, e.g.
                (n_rows, n_columns, n_bytes)

        """
        with hf.File(self.path, "r") as db:
            data_tuple = (
                db[data_path].shape[0],
                db[data_path].shape[1],
                db[data_path].nbytes,
            )

        return data_tuple

    def get_database_summary(self):
        """
        Get a summary of the database properties.

        Returns
        -------
        summary : list
                A list of properties that are in the database.

        """
        with hf.File(self.path, "r") as db:
            return list(db.keys())
