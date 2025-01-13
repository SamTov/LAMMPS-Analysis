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
Parent class for the project.
"""

from __future__ import annotations

import logging
import pathlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

from dot4dict import dotdict

import mdsuite.database.scheme as db
import mdsuite.file_io.file_read
from mdsuite.database.project_database import ProjectDatabase
from mdsuite.experiment import Experiment
from mdsuite.experiment.run import RunComputation
from mdsuite.utils import Units
from mdsuite.utils.helpers import NoneType

log = logging.getLogger(__name__)


class Project(ProjectDatabase):
    """Class for the main container of all experiments.

    The Project class acts as the encompassing class for analysis with MDSuite.
    It contains all method required to add and analyze new experiments. These
    experiments may then be compared with one another quickly. The state of the
    class is saved and updated after each operation in order to retain the
    most current state of the analysis.

    .. code-block:: python

        project = mdsuite.Project()
        project.add_experiment(
            name="NaCl",
            timestep=0.002,
            temperature=1400.0,
            units="metal",
            simulation_data="NaCl_gk_i_q.lammpstraj",
            active=False # calculations are only performed on active experiments
            )
        project.activate_experiments("NaCl") # set experiment state to active
        project.run.RadialDistributionFunction(number_of_configurations=500)
        project.disable_experiments("NaCl") # set experiment state to inactive

    Attributes
    ----------
    name : str
            The name of the project
    description : str
            A short description of the project
    storage_path : str
            Where to store the tensor_values and databases. This may not simply
            be the current directory if the databases are expected to be
            quite large.
    experiments : dict
            A dict of class objects. Class objects are instances of the experiment class
            for different experiments.

    """

    def __init__(
        self,
        name: str = None,
        storage_path: Union[str, Path] = "./",
        description: str = None,
    ):
        """Project class constructor.

        The constructor will check to see if the project already exists, if so,
        it will load the state of each of the classes so that they can be used
        again. If the project is new, the constructor will build the necessary
        file structure for the project.

        Parameters
        ----------
        name : str
                The name of the project.
        storage_path : str
                Where to store the tensor_values and databases. This should be
                a place with sufficient storage space for the full analysis.

        """
        super().__init__()
        if name is None:
            self.name = "MDSuite_Project"
        else:
            self.name = name
        self.storage_path = Path(storage_path).as_posix()

        # Properties
        self._experiments = {}

        # Check for project directory, if none exist, create a new one
        self.project_dir = Path(f"{self.storage_path}/{self.name}")

        if self.project_dir.exists():
            self.attach_file_logger()
            log.info("Loading the class state")
            log.info(f"Available experiments are: {self.db_experiments}")
        else:
            self.project_dir.mkdir(parents=True, exist_ok=True)
            self.attach_file_logger()
            log.info(f"Creating new project {self.name}")

        self.build_database()

        # Database Properties
        self.description = description

    def attach_file_logger(self):
        """Attach a file logger for this project."""
        logger = logging.getLogger("mdsuite")
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s (%(module)s): %(message)s"
        )
        # TODO this will potentially log two mds.Projects into the same file
        #   maybe there are some conditional logging Handlers that can check
        #   project.name, but for now this should be fine.
        channel = logging.FileHandler(self.project_dir / "mdsuite.log")
        channel.setLevel(logging.DEBUG)
        channel.setFormatter(formatter)

        logger.addHandler(channel)

    def __str__(self):
        r"""

        Returns
        -------
        str:
            A list of all available experiments like "1.) Exp01\n2.) Exp02\n3.) Exp03"

        """
        return "\n".join([f"{exp.id}.) {exp.name}" for exp in self.db_experiments])

    def add_experiment(
        self,
        name: str = NoneType,
        timestep: float = None,
        temperature: float = None,
        units: Union[str, Units] = None,
        cluster_mode: bool = None,
        active: bool = True,
        simulation_data: Union[
            str, pathlib.Path, mdsuite.file_io.file_read.FileProcessor, list
        ] = None,  # TODO make this the second argument, (name, data, ...)
        update_with_pubchempy: bool = False,
    ) -> Experiment:
        """Add an experiment to the project.

        Parameters
        ----------
        active: bool, default = True
                Activate the experiment when added
        cluster_mode : bool
                If true, cluster mode is parsed to the experiment class.
        name : str
                Name to use for the experiment.
        timestep : float
                Timestep used during the simulation.
        temperature : float
                Temperature the simulation was performed at and is to be used
                in calculation.
        units : str
                units used
        simulation_data:
            data that should be added to the experiment.
            see mdsuite.experiment.add_data() for details of the file specification.
            you can also create the experiment with simulation_data == None and add data
            later
        update_with_pubchempy : bool (default=False)
                Passed to the add data method.

        Notes
        -----
        Using custom NoneType to raise a custom ValueError message with useful info.

        Returns
        -------
        Experiment:
            The experiment object that was added to the project

        """
        if name is NoneType:
            raise ValueError(
                "Experiment name can not be empty! "
                "Use None to automatically generate a unique name."
            )

        if name is None:
            name = f"Experiment_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            # set the experiment name to the current date and time if None is provided

        # Run a query to see if that experiment already exists
        with self.session as ses:
            experiments = (
                ses.query(db.Experiment).filter(db.Experiment.name == name).all()
            )
        if len(experiments) > 0:
            log.info("This experiment already exists")
            self.load_experiments(name)
            return self.experiments[name]

        # If the experiment does not exists, instantiate a new Experiment
        new_experiment = Experiment(
            project=self,
            name=name,
            time_step=timestep,
            temperature=temperature,
            units=units,
            cluster_mode=cluster_mode,
        )

        new_experiment.active = active

        # Update the internal experiment dictionary for self.experiment property
        self._experiments[name] = new_experiment

        if simulation_data is not None:
            self.experiments[name].add_data(
                simulation_data, update_with_pubchempy=update_with_pubchempy
            )

        return self.experiments[name]

    def load_experiments(self, names: Union[str, list]):
        """Alias for activate_experiments."""
        self.activate_experiments(names)

    def activate_experiments(self, names: Union[str, list]):
        """Load experiments, such that they are used for the computations.

        Parameters
        ----------
        names: Name or list of names of experiments that should be instantiated
               and loaded into self.experiments.

        Returns
        -------
        Updates the class state.

        """
        if isinstance(names, str):
            names = [names]

        for name in names:
            self.experiments[name].active = True

    def disable_experiments(self, names: Union[str, list]):
        """Disable experiments.

        Parameters
        ----------
        names: Name or list of names of experiments that should be instantiated
               and loaded into self.experiments

        Returns
        -------

        """
        if isinstance(names, str):
            names = [names]

        for name in names:
            self.experiments[name].active = False

    def add_data(self, data_sets: dict):
        """Add simulation_data to a experiments.

        This is a method so that parallelization is
        possible amongst simulation_data addition to different experiments at the same
        time.

        Parameters
        ----------
        data_sets: dict
            keys: the names of the experiments
            values: str or mdsuite.file_io.file_read.FileProcessor
                refer to mdsuite.experiment.add_data() for an explanation of the file
                specification options

        Returns
        -------
        Updates the experiment classes.

        """
        for key, val in data_sets.items():
            self.experiments[key].add_data(val)

    @property
    def run(self) -> RunComputation:
        """Method to access the available calculators.

        Returns
        -------
        RunComputation:
            class that has all available calculators as properties

        """
        return RunComputation(experiments=list(self.active_experiments.values()))

    @property
    def experiments(self) -> Dict[str, Experiment]:
        """Get a DotDict of instantiated experiments!."""
        with self.session as ses:
            db_experiments = ses.query(db.Experiment).all()

        for exp in db_experiments:
            exp: db.Experiment
            if exp.name not in self._experiments:
                self._experiments[exp.name] = Experiment(project=self, name=exp.name)

        return dotdict(self._experiments)

    @property
    def active_experiments(self) -> Dict[str, Experiment]:
        """Get a DotDict of instantiated experiments that are currently selected!."""
        active_experiment = {
            key: val for key, val in self.experiments.items() if val.active
        }

        return dotdict(active_experiment)
