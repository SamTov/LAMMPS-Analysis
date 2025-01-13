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

import copy
import importlib.resources
import json
import logging
import pathlib
import typing
from pathlib import Path
from typing import List, Union

import numpy as np
import pubchempy as pcp

import mdsuite.file_io.extxyz_files
import mdsuite.file_io.file_read
import mdsuite.file_io.lammps_trajectory_files
import mdsuite.utils.meta_functions
from mdsuite.database.experiment_database import ExperimentDatabase
from mdsuite.database.simulation_database import (
    Database,
    SpeciesInfo,
    TrajectoryMetadata,
)
from mdsuite.experiment.run import RunComputation
from mdsuite.time_series import time_series_dict
from mdsuite.transformations import Transformations
from mdsuite.utils import config
from mdsuite.utils.exceptions import ElementMassAssignedZero
from mdsuite.utils.meta_functions import join_path
from mdsuite.utils.units import Units, units_dict

from .run_module import RunModule

log = logging.getLogger(__name__)


def _get_processor(simulation_data):
    """Read in one file."""
    if isinstance(simulation_data, str) or isinstance(simulation_data, pathlib.Path):
        suffix = pathlib.Path(simulation_data).suffix
        if suffix == ".lammpstraj":
            processor = mdsuite.file_io.lammps_trajectory_files.LAMMPSTrajectoryFile(
                simulation_data
            )
        elif suffix == ".extxyz":
            processor = mdsuite.file_io.extxyz_files.EXTXYZFile(simulation_data)
        else:
            raise ValueError(
                f"datafile ending '{suffix}' not recognized. If there is a reader for"
                " your file type, you will find it in mdsuite.file_io."
            )
    elif isinstance(simulation_data, mdsuite.file_io.file_read.FileProcessor):
        processor = simulation_data
    else:
        raise ValueError(
            "simulation_data must be either str, pathlib.Path or instance of"
            f" mdsuite.file_io.file_read.FileProcessor. Got '{type(simulation_data)}'"
            " instead"
        )

    return processor


class Experiment(ExperimentDatabase):
    """
    The central experiment class fundamental to all analysis.

    .. code-block:: python

        project = mdsuite.Project()
        project.add_experiment(
            name="NaCl",
            timestep=0.002,
            temperature=1400.0,
            units="metal",
            simulation_data="NaCl_gk_i_q.lammpstraj"
            )
        project.experiments.NaCl.run.RadialDistributionFunction(
            number_of_configurations=500
        )


    Attributes
    ----------
    name : str
            The name of the analysis being performed e.g. NaCl_1400K
    storage_path : str
            Path to where the tensor_values should be stored (best to have  drive
            capable of storing large files)
    temperature : float
            The temperature of the simulation that should be used in some analysis.
            Necessary as it cannot be easily read in from the simulation tensor_values.
    time_step : float
            Time step of the simulation e.g 0.002. Necessary as it cannot be easily
            read in from the trajectory.
    volume : float
            Volume of the simulation box
    species : dict
            A dictionary of the species in the experiment and their properties.
            Their properties includes index location in the trajectory file, mass of
            the species as taken from the PubChem database_path, and the charge taken
            from the same database_path. When using these properties, it is best that
            users confirm this information, with exception to the indices as they are
            read from the file and will be correct.
    number_of_atoms : int
            The total number of atoms in the simulation

    """

    def __init__(
        self,
        project,
        name,
        time_step=None,
        temperature=None,
        units: Union[str, Units] = None,
        cluster_mode=False,
    ):
        """
        Initialise the experiment class.

        Attributes
        ----------
        name : str
                The name of the analysis being performed e.g. NaCl_1400K
        temperature : float
                The temperature of the simulation that should be used in some analysis.
        time_step : float
                Time step of the simulation e.g 0.002. Necessary as it cannot be easily
                read in from the trajectory.
        units: Union[str, dict], default = "real"
            The units to be used in the experiment to convert to SI
        cluster_mode : bool
                If true, several parameters involved in plotting and parallelization
                will be adjusted so as to allow for optimal performance on a large
                computing cluster.

        """
        if not name[0].isalpha():
            raise ValueError(
                f"Experiment name must start with a letter! Found '{name[0]}' instead."
            )

        # Taken upon instantiation
        super().__init__(project=project, name=name)
        self.name = name
        self.storage_path = Path(project.storage_path, project.name).as_posix()
        self.cluster_mode = cluster_mode

        # ExperimentDatabase stored properties:
        # ------- #
        # set default values
        if self.number_of_configurations is None:
            self.number_of_configurations = 0
        # update database (None values are ignored)
        self.temperature = temperature
        self.time_step = time_step

        # Available properties that aren't set on default
        self.number_of_atoms = None
        # ------- #

        # Added from trajectory file

        if self.units is None:
            if units is None:
                units = mdsuite.units.REAL
            self.units = self.units_to_si(units)  # Units used during the simulation.

        self.box_array = None  # Box vectors.
        self.dimensions = None  # Dimensionality of the experiment.

        self.sample_rate = (
            None  # Rate at which configurations are dumped in the trajectory.
        )
        self.properties = None  # Properties measured in the simulation.
        self.property_groups = None  # Names of the properties measured in the simulation

        # Internal File paths
        self.path: Path
        self.database_path: Path
        self.figures_path: Path
        self._create_internal_file_paths()  # fill the path attributes

        # Check if the experiment exists and load if it does.
        self._load_or_build()

        self.analyse_time_series = RunModule(self, time_series_dict)

    @property
    def run(self) -> RunComputation:
        """Method to access the available calculators.

        Returns
        -------
        RunComputation:
            class that has all available calculators as properties

        """
        return RunComputation(experiment=self)

    def __repr__(self):
        """
        Representation of the class.

        In our case, the representation of the class is the name of the experiment.
        """
        return f"exp_{self.name}"

    def _create_internal_file_paths(self):
        """Create or update internal file paths.

        Attributes
        ----------
        path: Path
            The default path for the experiment files
        database_path: Path
            Path to the database, by default equal to self.path
        figures_path: Path
            Path to the figures directory

        """
        self.path = Path(self.storage_path, self.name)  # path to the experiment files
        self.database_path = self.path  # path to the databases
        self.figures_path = self.path / "figures"  # path to the figures directory

    def _build_model(self):
        """
        Build the 'experiment' for the analysis.

        A method to build the database_path in the hdf5 format. Within this method,
        several other are called to develop the database_path skeleton,
        get configurations, and process and store the configurations. The method is
        accompanied by a loading bar which should be customized to make it more
        interesting.
        """
        # Create new analysis directory and change into it
        try:
            self.path.mkdir()
            self.figures_path.mkdir()
            self.database_path.mkdir()
        except FileExistsError:  # throw exception if the file exits
            return

        # self.save_class()  # save the class state.
        log.info(f"** An experiment has been added titled {self.name} **")

    def cls_transformation_run(self, transformation: Transformations, *args, **kwargs):
        """Run the transformation.

        The Transformation class is updated with this experiment and afterwards
        performs the transformation.
        Preliminary work in accordance to https://github.com/zincware/MDSuite/issues/404

        Parameters
        ----------
        transformation: Transformations

        """
        transformation.experiment = self
        transformation.run_transformation(*args, **kwargs)

    @staticmethod
    def units_to_si(units_system) -> Units:
        """
        Returns a dictionary with equivalences from the unit experiment given by a
        string to SI. Along with some constants in the unit experiment provided
        (boltzmann, or other conversions). Instead, the user may provide a dictionary.
        In that case, the dictionary will be used as the unit experiment.


        Parameters
        ----------
        units_system (str) -- current unit experiment
        dimension (str) -- dimension you would like to change

        Returns
        -------
        units: Units
            dataclass that contains the conversion factors to SI

        """
        if isinstance(units_system, Units):
            return units_system
        elif isinstance(units_system, str):
            try:
                units = units_dict[units_system]
            except KeyError:
                raise KeyError(
                    f"The unit '{units_system}' is not implemented."
                    f" The available units are: {list(units_dict)}"
                )
        else:
            raise ValueError(
                "units has to be of type Units or str,"
                f" found {type(units_system)} instead"
            )
        return units

    def _load_or_build(self) -> bool:
        """
        Check if the experiment already exists and decide whether to load it or build a
        new one.
        """
        # Check if the experiment exists and load if it does.
        if Path(self.path).exists():
            log.debug(
                f"This experiment ({self.name}) already exists! I'll load it up now."
            )
            return True
        else:
            log.info(f"Creating a new experiment ({self.name})!")
            self._build_model()
            return False

    def run_visualization(
        self,
        species: list = None,
        molecules: bool = False,
        unwrapped: bool = False,
    ):
        """
        Perform a trajectory visualization.

        Parameters
        ----------
        species : list
                A list of species of molecules to study.
        molecules : bool
                If true, molecule groups will be used.
        unwrapped : bool
                If true, unwrapped coordinates will be visualized.

        Returns
        -------
        Displays a visualization app.

        """
        import_error_msg = (
            "It looks like you don't have the necessary plugin for "
            "the visualizer extension. Please install znvis with"
            " pip install znvis in order to use the MDSuite visualizer."
        )
        try:
            from mdsuite.visualizer.znvis_visualizer import SimulationVisualizer
        except ImportError:
            log.info(import_error_msg)
            return

        if molecules:
            if species is None:
                species = list(self.molecules)
        if species is None:
            species = list(self.species)

        if config.jupyter:
            log.info(
                "ZnVis visualizer currently does not support deployment from "
                "jupyter. Please run your analysis as a python script to use"
                "the visualizer."
            )
            return
        else:
            visualizer = SimulationVisualizer(
                species=species, unwrapped=unwrapped, database_path=self.database_path
            )
            visualizer.run_visualization()

    # def map_elements(self, mapping: dict = None):
    #     """
    #     Map numerical keys to element names in the Experiment class and database_path.
    #
    #     Returns
    #     -------
    #     Updates the class
    #     """
    #
    #     if mapping is None:
    #         log.info("Must provide a mapping")
    #         return
    #
    #     # rename keys in species dictionary
    #     for item in mapping:
    #         self.species[mapping[item]] = self.species.pop(item)
    #
    #     # rename database_path groups
    #     db_object = Database(name=os.path.join(self.database_path,
    #     "database_path.hdf5"))
    #     db_object.change_key_names(mapping)
    #
    #     self.save_class()  # update the class state
    #
    #
    # def set_element(self, old_name, new_name):
    #     """
    #     Change the name of the element in the self.species dictionary
    #
    #     Parameters
    #     ----------
    #     old_name : str
    #             Name of the element you want to change
    #     new_name : str
    #             New name of the element
    #     """
    #     # Check if the new name is new
    #     if new_name != old_name:
    #         self.species[new_name] = self.species[old_name]  # update dict
    #         del self.species[old_name]  # remove old entry

    def set_charge(self, element: str, charge: float):
        """
        Set the charge/s of an element.

        Parameters
        ----------
        element : str
                Name of the element whose charge you want to change
        charge : list
                New charge/s of the element

        """
        species = self.species
        species[element].charge = [charge]
        self.species = species

    def set_mass(self, element: str, mass: float):
        """
        Set the mass/es of an element.

        Parameters
        ----------
        element : str
                Name of the element whose mass you want to change
        mass : list
                New mass/es of the element

        """
        species = self.species
        species[element].mass = mass
        self.species = species

    def add_data(
        self,
        simulation_data: Union[
            str, pathlib.Path, mdsuite.file_io.file_read.FileProcessor, list
        ],
        force: bool = False,
        update_with_pubchempy: bool = False,
    ):
        """
        Add data to experiment. This method takes a filename, file path or a file
        reader (or a list thereof). If given a filename, it will try to instantiate the
        appropriate file reader with its default arguments. If you have a custom data
        format with its own reader or want to use non-default arguments for your reader,
        instantiate the reader and pass it to this method.
        TODO reference online documentation of data loading in the error messages

        Parameters
        ----------
        simulation_data : str or pathlib.Path or mdsuite.file_io.file_read.FileProcessor
            or list thereof
            if str or pathlib.Path: path to the file that contains the simulation_data
            if mdsuite.file_io.file_read.FileProcessor: An already instantiated file
            reader from mdsuite.file_io
            if list : must be list of any of the above (can be mixed).
        force : bool
            If true, a file will be read regardless of if it has already been seen.
            Default: False
        update_with_pubchempy: bool
            Whether or not to look for the masses of the species in pubchempy.
            Default: True.

        """
        if isinstance(simulation_data, list):
            for elem in simulation_data:
                proc = _get_processor(elem)
                self._add_data_from_file_processor(
                    proc, force=force, update_with_pubchempy=update_with_pubchempy
                )
        else:
            proc = _get_processor(simulation_data)
            self._add_data_from_file_processor(
                proc, force=force, update_with_pubchempy=update_with_pubchempy
            )

    def _add_data_from_file_processor(
        self,
        file_processor: mdsuite.file_io.file_read.FileProcessor,
        force: bool = False,
        update_with_pubchempy: bool = False,
    ):
        """
        Add tensor_values to the database_path.

        Parameters
        ----------
        file_processor
            The FileProcessor that is able to provide the metadata and the trajectory
            to be saved
        force : bool
                If true, a file will be read regardless of if it has already
                been seen.
        update_with_pubchempy: bool
                Whether or not to look for the masses of the species in pubchempy

        """
        already_read = str(file_processor) in self.read_files
        if already_read and not force:
            log.info(
                "This file has already been read, skipping this now."
                "If this is not desired, please add force=True "
                "to the command."
            )
            return

        database = Database(self.database_path / "database.hdf5")

        metadata = file_processor.metadata
        architecture = _species_list_to_architecture_dict(
            metadata.species_list, metadata.n_configurations
        )
        if not database.database_exists():
            self._store_metadata(metadata, update_with_pubchempy=update_with_pubchempy)
            database.initialize_database(architecture)
        else:
            database.resize_datasets(architecture)

        for i, batch in enumerate(file_processor.get_configurations_generator()):
            database.add_data(chunk=batch)
            self.number_of_configurations += batch.chunk_size

        self.version += 1

        self.memory_requirements = database.get_memory_information()

        # set at the end, because if something fails, the file was not properly read.
        self.read_files = self.read_files + [str(file_processor)]

    def load_matrix(
        self,
        property_name: str = None,
        species: typing.Iterable[str] = None,
        select_slice: np.s_ = None,
        path: typing.Iterable[str] = None,
    ):
        """
        Load a desired property matrix.

        Parameters
        ----------
        property_name : str
                Name of the matrix to be loaded, e.g. 'Unwrapped_Positions',
                'Velocities'
        species : Iterable[str]
                List of species to be loaded
        select_slice : np.slice
                A slice to select from the database_path.
        path : str
                optional path to the database_path.

        Returns
        -------
        property_matrix : np.array, tf.Tensor
                Tensor of the property to be studied. Format depends on kwargs.

        """
        database = Database(self.database_path / "database.hdf5")

        if path is not None:
            return database.load_data(path_list=path, select_slice=select_slice)

        else:
            # If no species list is given, use all species in the Experiment.
            if species is None:
                species = list(self.species.keys())
            # If no slice is given, load all configurations.
            if select_slice is None:
                select_slice = np.s_[:]  # set the numpy slice object.

        path_list = []
        for item in species:
            path_list.append(join_path(item, property_name))
        return database.load_data(path_list=path_list, select_slice=select_slice)

    def _store_metadata(self, metadata: TrajectoryMetadata, update_with_pubchempy=False):
        """Save Metadata in the SQL DB.

        Parameters
        ----------
        metadata: TrajectoryMetadata
        update_with_pubchempy: bool
            Load data from pubchempy and add it to fill missing infomration

        """
        # new trajectory: store all metadata and construct a new database
        self.temperature = metadata.temperature
        self.box_array = metadata.box_l
        if self.box_array is not None:
            self.dimensions = mdsuite.utils.meta_functions.get_dimensionality(
                self.box_array
            )
        else:
            self.dimensions = None
        # todo look into replacing these properties
        self.sample_rate = metadata.sample_rate
        species_list = copy.deepcopy(metadata.species_list)
        if update_with_pubchempy:
            update_species_attributes_with_pubchempy(species_list)
        # store the species information in dict format
        species_dict = {}
        for sp_info in species_list:
            species_dict[sp_info.name] = {
                # look here
                "mass": sp_info.mass,
                "charge": sp_info.charge,
                "n_particles": sp_info.n_particles,
                # legacy: calculators use this list to determine the number of particles
                # TODO change this.
                "indices": list(range(sp_info.n_particles)),
                "properties": [prop_info.name for prop_info in sp_info.properties],
            }
        self.species = species_dict
        # assume the same property for each species
        self.property_groups = next(iter(species_dict.values()))["properties"]
        # update n_atoms
        self.number_of_atoms = sum(sp["n_particles"] for sp in species_dict.values())


def update_species_attributes_with_pubchempy(species_list: List[SpeciesInfo]):
    """
    Add information to the species dictionary.

    A fundamental part of this package is species specific analysis. Therefore, the
    Pubchempy package is used to add important species specific information to the
    class. This will include the charge of the ions which will be used in
    conductivity calculations.

    """
    with importlib.resources.open_text(
        "mdsuite.data", "PubChemElements_all.json"
    ) as json_file:
        pse = json.loads(json_file.read())

    # Try to get the species tensor_values from the Periodic System of Elements file

    # set/find masses
    for sp_info in species_list:
        for entry in pse:
            if pse[entry][1] == sp_info.name:
                sp_info.mass = [float(pse[entry][3])]

                # If gathering the tensor_values from the PSE file was not successful
    # try to get it from Pubchem via pubchempy
    for sp_info in species_list:
        if sp_info.mass is None:
            try:
                temp = pcp.get_compounds(sp_info.name, "name")
                temp[0].to_dict(
                    properties=[
                        "atoms",
                        "bonds",
                        "exact_mass",
                        "molecular_weight",
                        "elements",
                    ]
                )
                sp_info.mass = [temp[0].molecular_weight]
                log.debug(temp[0].exact_mass)
            except (ElementMassAssignedZero, IndexError):
                sp_info.mass = 0.0
                log.warning(f"WARNING element {sp_info.name} has been assigned mass=0.0")
    return species_list


def _species_list_to_architecture_dict(species_list, n_configurations):
    # TODO let the database handler use the species list directly instead of the dict
    """
    Converter from species list to legacy architecture dict

    Parameters
    ----------
    species_list
    n_configurations.

    Returns
    -------
    dict like architecture = {'Na':{'Positions':(n_part, n_config, n_dim)}}

    """
    architecture = {}
    for sp_info in species_list:
        architecture[sp_info.name] = {}
        for prop_info in sp_info.properties:
            architecture[sp_info.name][prop_info.name] = (
                sp_info.n_particles,
                n_configurations,
                prop_info.n_dims,
            )
    return architecture
