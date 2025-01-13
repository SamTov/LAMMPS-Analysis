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

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship

from .types import JSONEncodedDict, MutableDict

log = logging.getLogger(__name__)

Base = declarative_base()


class SpeciesAssociation(Base):
    """Connection between Computations and Experiment Species.

    This table is required to be defined specifically, because we need add the count,
    e.g. Na - Na would otherwise only appear as Na.
    """

    __tablename__ = "species_association"
    computation_results_id = Column(
        ForeignKey("computation_results.id"), primary_key=True
    )
    experiment_species_id = Column(ForeignKey("experiment_species.id"), primary_key=True)

    count = Column(
        Integer, default=1
    )  # how often a species occurs, e.g. Na - Na - Cl ADF would be 2, 1

    computation_result = relationship("ComputationResult", back_populates="species")
    species = relationship("ExperimentSpecies", back_populates="computation_result")

    @property
    def name(self):
        """Get the name of the species."""
        return self.species.name


class Project(Base):
    """Project table."""

    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    description = Column(String, nullable=True)

    experiments = relationship("Experiment")


class Experiment(Base):
    """Class for the experiment table associated with the Project table."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True)
    name = Column(String)

    active = Column(Boolean, default=False)
    # Whether this experiment is currently loaded in the project class

    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"))
    project = relationship("Project")

    experiment_attributes = relationship(
        "ExperimentAttribute", cascade="all, delete", back_populates="experiment"
    )

    computations = relationship("Computation")

    species = relationship("ExperimentSpecies")

    def __repr__(self):
        """
        Representation of the experiment table.

        Returns
        -------
        information : str
                Experiment number and name as an fstring

        """
        return f"{self.id}: {self.name}"

    def get_species(self) -> dict:
        """Get the species information for the experiment."""
        species_dict = {}
        for species in self.species:
            if species.molecule:
                continue
            species: ExperimentSpecies
            species_dict[species.name] = species.data

        return species_dict

    def get_molecules(self) -> dict:
        """Get the molecules information for the experiment."""
        molecule_dict = {}
        for molecule in self.species:
            if molecule.molecule:
                molecule: ExperimentSpecies
                molecule_name = molecule.name
                molecule_dict[molecule_name] = molecule.data

        return molecule_dict


class ExperimentAttribute(Base):
    """
    Class for the experiment data table.

    This table is arbitrarily defined and therefore anything can be added to it.

    Attributes
    ----------
    id : int
            Unique identifier of the row.
    name : str
            name of the property being recorded.
    value : float
            numeric value of the property.
    str_value : str
            String value of the property.

    """

    __tablename__ = "experiment_attributes"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    data = Column(MutableDict.as_mutable(JSONEncodedDict))

    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"))
    experiment = relationship("Experiment", back_populates="experiment_attributes")

    def __repr__(self):
        if self.value is not None:
            return f"{self.value}"
        elif self.str_value is not None:
            return self.str_value
        else:
            return f"{self.name}"


class ExperimentSpecies(Base):
    """Table for storing species information.

    This table is used to store species and molecule information that can be related to
    a specific experiment

    """

    # TODO this could potentially be replaced by ExperimentAttribute

    __tablename__ = "experiment_species"
    id = Column(Integer, primary_key=True)

    name = Column(String)
    data = Column(MutableDict.as_mutable(JSONEncodedDict))
    molecule = Column(Boolean, default=False)

    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"))
    experiment = relationship("Experiment", back_populates="species")

    computation_result = relationship("SpeciesAssociation", back_populates="species")

    def __repr__(self):
        return f"{self.name}_obj"


class Computation(Base):
    """Class for the computation table."""

    __tablename__ = "computations"

    id = Column(Integer, primary_key=True)
    name = Column(String, default="Computation")

    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"))
    experiment = relationship("Experiment")

    computation_attributes = relationship(
        "ComputationAttribute", cascade="all, delete", back_populates="computation"
    )
    computation_results = relationship(
        "ComputationResult",
        cascade="all, delete",
        back_populates="computation",
        lazy=True,
    )

    def __repr__(self):
        """
        Representation of the experiment table.

        Returns
        -------
        information : str
                Experiment number and name as an fstring

        """
        return f"Exp{self.experiment_id}_{self.name}_{self.id}"

    @property
    def data_dict(self) -> dict:
        """

        Returns
        -------
        species_dict: dict
            A dictionary of the type
            {
                Li:
                    {
                        a: 1.2,
                        uncert: 0.1,
                        time: [1, 2, 3, ],
                        msd: [0.1, 0.3, 0.7]
                    },
                Cl:
                    {
                        a: 1.2,
                        uncert: 0.1,
                        time: [1, 2, 3, ],
                        msd: [0.1, 0.3, 0.7]
                    },
            }
            where the keys are defined by species (multiple species are joined by "_")
            and the dimension argument of the computation_data

        """
        species_dict = {}
        for result in self.computation_results:
            result: ComputationResult
            species_keys_list = []
            for species_associate in result.species:
                species_associate: SpeciesAssociation
                species_keys_list += species_associate.count * [
                    species_associate.species.name
                ]
            species_keys = "_".join(species_keys_list)
            if species_keys == "":
                species_keys = "System"
            # iterating over associates
            species_dict[species_keys] = result.data

        return species_dict

    def __getitem__(self, item):
        """Allow for subscription.

        Parameters
        ----------
        item: str
            The key of self.data_dict to access

        Returns
        -------
        The value inside self.data_dict[item], usually a dict or single value.
            raises a KeyError if the given key is not available

        Examples
        --------
        >>> Computation["Na"]
        instead of
        >>> Computation.data_dict["Na"]


        """
        try:
            return self.data_dict[item]
        except KeyError:
            raise KeyError(
                f"Could not find {item} - available keys are {self.data_dict.keys()}"
            )

    def keys(self) -> list:
        """Map the data_dict keys."""
        return list(self.data_dict.keys())

    @property
    def computation_parameter(self) -> dict:
        """Get a dict of all used computation parameters.

        Examples
        --------
        The following example is taken from the RDF calculator
        {
            "number_of_bins": null,
            "number_of_configurations": 100,
            "correlation_time": 1,
            "atom_selection": "slice(None, None, None)",
            "data_range": 1,
            "cutoff": null,
            "start": 0,
            "stop": null,
            "species": null,
            "molecules": false,
            "version": 1
        }

        """
        computation_parameter = {}
        for comp_attr in self.computation_attributes:
            computation_parameter[comp_attr.name] = comp_attr.data["serialized_value"]
        return computation_parameter

    @property
    def data_range(self) -> int:
        """Get the data_range stored in computation_attributes."""
        for comp_attr in self.computation_attributes:
            if comp_attr.name == "data_range":
                return int(comp_attr.data["serialized_value"])

    @property
    def subjects(self) -> list:
        """Get the subjects stored in computation_attributes."""
        log.warning("This function will be depreciated!")
        subjects = []
        for x in self.computation_results:
            subjects.append(x.species.species.name)
        return subjects


class ComputationAttribute(Base):
    """Class for the meta data of a computation."""

    __tablename__ = "computation_attributes"

    # Table data
    id = Column(Integer, primary_key=True)
    name = Column(String)
    data = Column(MutableDict.as_mutable(JSONEncodedDict))

    # Relation data
    computation_id = Column(Integer, ForeignKey("computations.id", ondelete="CASCADE"))
    computation = relationship("Computation", back_populates="computation_attributes")

    def __repr__(self):
        return f"{self.name}: {self.value} - {self.str_value}"


class ComputationResult(Base):
    """raw computation data of a calculation."""

    __tablename__ = "computation_results"

    id = Column(Integer, primary_key=True)

    data = Column(MutableDict.as_mutable(JSONEncodedDict))

    # Relation data
    computation_id = Column(Integer, ForeignKey("computations.id", ondelete="CASCADE"))
    computation = relationship("Computation", back_populates="computation_results")

    # Many <-> Many
    species = relationship("SpeciesAssociation", back_populates="computation_result")
