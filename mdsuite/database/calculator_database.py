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
Code related to the calculator database.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, List

from sqlalchemy import and_

import mdsuite.database.scheme as db
from mdsuite.utils.meta_functions import is_jsonable

if TYPE_CHECKING:
    from mdsuite.experiment import Experiment

log = logging.getLogger(__name__)


@dataclass
class StoredParameters:
    """Dummy Class for type hinting"""

    pass


@dataclass
class ComputationResults:
    """
    A wrapper class for the results of a computation.

    This class is returned when data is loaded from the SQL database.
    """

    data: dict = field(default_factory=dict)
    subjects: dict = field(default_factory=list)


def convert_to_db(val):
    """Convert the given value to something that can be stored in the database"""
    if is_jsonable(val):
        if not isinstance(val, dict):
            val = {"serialized_value": val}
    else:
        val = {"serialized_value": str(val)}
    return val


class CalculatorDatabase:
    """Database Interactions of the calculator class

    This class handles the interaction of the calculator with the project database
    """

    def __init__(
        self, experiment, stored_parameters: StoredParameters, analysis_name: str
    ):
        """
        Constructor for the calculator database

        Parameters
        ----------
        experiment : Experiment
                Experiment for which this calculation is being performed.
        stored_parameters : StoredParameters
                Parameters of the calculator to be looked for and stored in the
                SQL database,
        analysis_name : str
                Name of the analysis being performed.
        """
        self.experiment: Experiment = experiment
        self.db_computation: db.Computation = None
        self.database_group = None
        self.analysis_name = analysis_name
        self.load_data = None

        self.stored_parameters = stored_parameters

        self._queued_data = []

        # List of computation attributes that will be added to the database
        self.db_computation_attributes = []

    def prepare_db_entry(self):
        """
        Prepare a database entry based on the attributes defined in the init

        Returns
        -------
        Adds computation information to the SQL database.
        """
        with self.experiment.project.session as ses:
            experiment = (
                ses.query(db.Experiment)
                .filter(db.Experiment.name == self.experiment.name)
                .first()
            )

        self.db_computation = db.Computation(experiment=experiment)
        self.db_computation.name = self.analysis_name

    def get_computation_data(self) -> db.Computation:
        """Query the database for computation data

        This method used the self.args dataclass to look for matching
        calculator attributes and returns a db.Computation object if
        the calculation has already been performed

        Return
        ------
        db.Computation
            Returns the computation object from the database if available,
            otherwise returns None
        """
        log.debug(
            f"Getting data for {self.experiment.name}, computation"
            f" {self.analysis_name} with args {self.stored_parameters}"
        )
        with self.experiment.project.session as ses:
            experiment = (
                ses.query(db.Experiment)
                .filter(db.Experiment.name == self.experiment.name)
                .first()
            )

            #  filter the correct experiment
            computations = ses.query(db.Computation).filter(
                db.Computation.experiment == experiment,
                db.Computation.name == self.analysis_name,
            )

            # filter set args
            for args_field in fields(self.stored_parameters):
                key = args_field.name
                val = getattr(self.stored_parameters, key)
                computations = computations.filter(
                    db.Computation.computation_attributes.any(
                        and_(
                            db.ComputationAttribute.name == key,
                            db.ComputationAttribute.data == convert_to_db(val),
                        )
                    )
                )

            # filter the version of the experiment, e.g. run new computation
            # if the experiment version has changed
            computations = computations.filter(
                db.Computation.computation_attributes.any(
                    and_(
                        db.ComputationAttribute.name == "version",
                        db.ComputationAttribute.data
                        == convert_to_db(self.experiment.version),
                    )
                )
            )

            computations = computations.all()
            if len(computations) > 0:
                log.debug("Calculation already performed! Loading it up")
            # loading data_dict to avoid DetachedInstance errors
            # this can take some time, depending on the size of the data
            # TODO remove and use lazy call
            for computation in computations:
                _ = computation.data_dict
                _ = computation.data_range
        if len(computations) > 0:
            if len(computations) > 1:
                log.warning(
                    "Something went wrong! Found more than one computation with the"
                    " given arguments!"
                )
            return computations[0]  # it should only be one value

        return None

    def save_computation_args(self):
        """Store the user args

        This method stored the user args from the self.args dataclass
        into SQLAlchemy objects and adds them to a list which will be
        written to the database after the calculation was successful.
        """
        for args_field in fields(self.stored_parameters):
            key = args_field.name
            val = getattr(self.stored_parameters, key)
            computation_attribute = db.ComputationAttribute(
                name=key, data=convert_to_db(val)
            )

            self.db_computation_attributes.append(computation_attribute)

        # save the current experiment version in the ComputationAttributes
        experiment_version = db.ComputationAttribute(
            name="version", data=convert_to_db(self.experiment.version)
        )
        self.db_computation_attributes.append(experiment_version)

    def save_db_data(self, staged_data: List[ComputationResults]):
        """Save all the collected computation attributes and computation data to the
        database

        This will be run after the computation was successful.
        """
        with self.experiment.project.session as ses:
            ses.add(self.db_computation)
            for val in self.db_computation_attributes:
                # I need to set the relation inside the session.
                val.computation = self.db_computation
                ses.add(val)

            for data_obj in staged_data:
                # TODO consider renaming species to e.g., subjects, because species here
                #  can also be molecules
                # data_obj: ComputationResults
                computation_result = db.ComputationResult(
                    computation=self.db_computation, data=data_obj.data
                )
                species_list = []
                for species in data_obj.subjects:
                    # this will collect duplicates that can be counted later,
                    # otherwise I would use .in_
                    species_list.append(
                        ses.query(db.ExperimentSpecies)
                        .filter(db.ExperimentSpecies.name == species)
                        .first()
                    )
                # in case of e.g. `System` species will be [None], which is then removed
                species_list = [x for x in species_list if x is not None]
                for species, count in Counter(species_list).items():
                    associate = db.SpeciesAssociation(count=count)
                    associate.species = species
                    computation_result.species.append(associate)

                ses.add(computation_result)

            ses.commit()
