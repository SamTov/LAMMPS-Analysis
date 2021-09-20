"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the experiment database.

This class does contain all properties that are written / read from the database

"""
from __future__ import annotations

import logging

import mdsuite.database.scheme as db
from mdsuite.database.scheme import Project, Experiment, ExperimentAttribute
from mdsuite.utils.database import get_or_create
import pandas as pd

from pathlib import Path
from typing import TYPE_CHECKING, List, Dict

if TYPE_CHECKING:
    from mdsuite import Project

log = logging.getLogger(__name__)


class ExperimentDatabase:
    def __init__(self, project: Project, experiment_name):
        self.project = project
        self.name = experiment_name

    def export_property_data(self, parameters: dict) -> List[db.Computation]:
        """
        Export property data from the SQL database.

        Parameters
        ----------
        parameters : dict
                Parameters to be used in the addition, i.e.
                {"Analysis": "Green_Kubo_Self_Diffusion", "Subject": "Na", "data_range": 500}
        Returns
        -------
        output : list
                A list of rows represented as dictionaries.
        """
        with self.project.session as ses:
            subjects = parameters.pop('subjects', None)
            experiment = parameters.pop('experiment', None)

            query = ses.query(db.Computation)
            for key, val in parameters.items():
                if isinstance(val, str):
                    query = query.filter(
                        db.Computation.computation_attributes.any(name=key),
                        db.Computation.computation_attributes.any(str_value=val)
                    )
                else:
                    query = query.filter(
                        db.Computation.computation_attributes.any(name=key),
                        db.Computation.computation_attributes.any(value=val)
                    )
            if experiment is not None:
                query = query.filter(db.Computation.experiment.has(name=experiment))
            computations_all_subjects = query.all()

            # Filter out subjects, this is easier to do this way around than via SQL statements (feel free to rewrite!)
            computations = []
            if subjects is not None:
                for x in computations_all_subjects:
                    if set(x.subjects).issubset(subjects):
                        computations.append(x)
            else:
                computations = computations_all_subjects

            for computation in computations:
                _ = computation.data_dict

        return computations

    @property
    def active(self):
        """Get the state (activated or not) of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
        return experiment.active

    @active.setter
    def active(self, value):
        """Set the state (activated or not) of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            experiment.active = value
            ses.commit()

    @property
    def temperature(self):
        """Get the temperature of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            temperature = ses.query(ExperimentAttribute).filter(ExperimentAttribute.experiment == experiment).filter(
                ExperimentAttribute.name == "temperature").first()
        try:
            return temperature.value
        except AttributeError:
            return None

    @temperature.setter
    def temperature(self, value):
        """Set the temperature of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            temperature: ExperimentAttribute = get_or_create(ses, ExperimentAttribute, experiment=experiment,
                                                             name="temperature")
            temperature.value = value
            ses.commit()

    @property
    def time_step(self):
        """Get the time_step of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            time_step = ses.query(ExperimentAttribute).filter(ExperimentAttribute.experiment == experiment).filter(
                ExperimentAttribute.name == "time_step").first()
        try:
            return time_step.value
        except AttributeError:
            return None

    @time_step.setter
    def time_step(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            time_step: ExperimentAttribute = get_or_create(ses, ExperimentAttribute, experiment=experiment,
                                                           name="time_step")
            time_step.value = value
            ses.commit()

    @property
    def units(self) -> Dict[str, float]:
        """Get the units of the experiment"""
        units = {}
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
            for experiment_data in experiment.experiment_attributes:
                if experiment_data.name.startswith('unit_system_'):
                    unit_name = experiment_data.name.split('unit_system_', 1)[1]
                    # everything after, max 1 split
                    units[unit_name] = experiment_data.value

        return units

    @units.setter
    def units(self, value: dict):
        """Set the units of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            for unit in value:
                unit_entry = get_or_create(ses, ExperimentAttribute, experiment=experiment, name=f"unit_system_{unit}")
                unit_entry.value = value[unit]

            ses.commit()

    @property
    def number_of_configurations(self) -> int:
        """Get the time_step of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            number_of_configurations = ses.query(ExperimentAttribute).filter(
                ExperimentAttribute.experiment == experiment).filter(
                ExperimentAttribute.name == "number_of_configurations").first()
        try:
            return int(number_of_configurations.value)
        except AttributeError:
            return None

    @number_of_configurations.setter
    def number_of_configurations(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            number_of_configurations: ExperimentAttribute = get_or_create(ses, ExperimentAttribute,
                                                                          experiment=experiment,
                                                                          name="number_of_configurations")
            number_of_configurations.value = value
            ses.commit()

    @property
    def number_of_atoms(self) -> int:
        """Get the time_step of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            number_of_atoms = ses.query(ExperimentAttribute).filter(
                ExperimentAttribute.experiment == experiment).filter(
                ExperimentAttribute.name == "number_of_atoms").first()
        try:
            return int(number_of_atoms.value)
        except AttributeError:
            return None

    @number_of_atoms.setter
    def number_of_atoms(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            number_of_atoms: ExperimentAttribute = get_or_create(ses, ExperimentAttribute, experiment=experiment,
                                                                 name="number_of_atoms")
            number_of_atoms.value = value
            ses.commit()

    @property
    def sample_rate(self):
        """Get the sample_rate of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            sample_rate = ses.query(ExperimentAttribute).filter(ExperimentAttribute.experiment == experiment).filter(
                ExperimentAttribute.name == "sample_rate").first()
        try:
            return sample_rate.value
        except AttributeError:
            return None

    @sample_rate.setter
    def sample_rate(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            sample_rate: ExperimentAttribute = get_or_create(ses, ExperimentAttribute, experiment=experiment,
                                                             name="sample_rate")
            sample_rate.value = value
            ses.commit()

    @property
    def volume(self):
        """Get the volume of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            volume = ses.query(ExperimentAttribute).filter(ExperimentAttribute.experiment == experiment).filter(
                ExperimentAttribute.name == "volume").first()
        try:
            return volume.value
        except AttributeError:
            return None

    @volume.setter
    def volume(self, value):
        """Set the volume of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            volume: ExperimentAttribute = get_or_create(ses, ExperimentAttribute, experiment=experiment, name="volume")
            volume.value = value
            ses.commit()

    @property
    def box_array(self):
        """Get the sample_rate of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            box_arrays = ses.query(ExperimentAttribute).filter(ExperimentAttribute.experiment == experiment).filter(
                ExperimentAttribute.name.startswith("box_array")).all()

            box_array = [box_side.value for box_side in box_arrays]

        return box_array

    @box_array.setter
    def box_array(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            for idx, box_length in enumerate(value):
                sample_rate: ExperimentAttribute = get_or_create(
                    ses, ExperimentAttribute, experiment=experiment, name=f"box_array_{idx}")
                sample_rate.value = box_length
            ses.commit()

    @property
    def species(self):
        """Get species
        """
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
            species_dict = experiment.species

        return species_dict

    @species.setter
    def species(self, value):
        """

        Parameters
        ----------
        value

        Notes
        -----

        species = {C: {indices: [1, 2, 3], mass: [12.0], charge: [0]}}

        """
        if value is None:
            return
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
            for species_name in value:
                species = ses.query(db.ExperimentAttribute).filter(
                    db.ExperimentAttribute.name == "species").join(
                    db.ExperimentAttribute.experiment_attribute_lists).filter(
                    db.ExperimentAttributeList.name == species_name).first()

                if species is None:
                    log.debug(f"Creating new species db entry for {species_name}.")
                    species = db.ExperimentAttribute(experiment=experiment, name="species")
                    db_species_name = db.ExperimentAttributeList(experiment_attribute=species, name="name",
                                                                 str_value=species_name)
                    ses.add(species)
                    ses.add(db_species_name)

                for species_attr, species_values in value[species_name].items():
                    try:
                        for species_value in species_values:
                            x = db.ExperimentAttributeList(experiment_attribute=species, name=species_attr,
                                                           value=species_value)
                            ses.add(x)

                    except TypeError:
                        # e.g., float or int values that are not iterable
                        if species_values is not None:
                            log.warning(f"Updating {species_attr} with {species_values}")
                            x = db.ExperimentAttributeList(experiment_attribute=species, name=species_attr,
                                                           value=species_values)
                            ses.add(x)
            ses.commit()

    @property
    def property_groups(self):
        """Get the property groups from the database

        Returns
        -------
        property_groups: dict
                Example: {'Positions': [3, 4, 5], 'Velocities': [6, 7, 8], ...}
        """
        property_groups = {}
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
            for experiment_data in experiment.experiment_attributes:
                if experiment_data.name.startswith('property_group_'):
                    property_group = experiment_data.name.split('property_group_', 1)[1]
                    # everything after, max 1 split
                    group_values = property_groups.get(property_group, [])
                    group_values.append(int(experiment_data.value))
                    property_groups[property_group] = group_values

        return property_groups

    @property_groups.setter
    def property_groups(self, value):
        """Write the property groups to the database"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
            for group in value:
                for group_value in value[group]:
                    get_or_create(ses, ExperimentAttribute, name=f"property_group_{group}", value=group_value,
                                  experiment=experiment)

            ses.commit()

    @property
    def read_files(self):
        """

        Returns
        -------
        read_files: list[Path]
            A List of all files that were added to the database already

        """
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
            read_files = ses.query(ExperimentAttribute).filter(ExperimentAttribute.experiment == experiment).filter(
                ExperimentAttribute.name == "read_file").all()
            read_files = [Path(file.str_value) for file in read_files]
        return read_files

    @read_files.setter
    def read_files(self, value):
        """Add a file that has been read to the database

        Does nothing if the file already  exists within the database

        Parameters
        ----------
        value: str, Path
            A filepath that will be added to the database

        """
        if value is None:
            return
        if isinstance(value, Path):
            value = value.as_posix()
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
            get_or_create(ses, ExperimentAttribute, name="read_file", str_value=value, experiment=experiment)
            ses.commit()

    @property
    def radial_distribution_function_state(self) -> bool:
        """Get the radial_distribution_function_state of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            rdf_state = ses.query(ExperimentAttribute).filter(ExperimentAttribute.experiment == experiment).filter(
                ExperimentAttribute.name == "radial_distribution_function_state").first()
        try:
            return rdf_state.value
        except AttributeError:
            return False

    @radial_distribution_function_state.setter
    def radial_distribution_function_state(self, value):
        """Set the radial_distribution_function_state of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            rdf_state: ExperimentAttribute = get_or_create(ses, ExperimentAttribute, experiment=experiment,
                                                           name="radial_distribution_function_state")
            rdf_state.value = value
            ses.commit()

    @property
    def simulation_data(self) -> dict:
        """
        Load simulation data from internals.
        If not available try to read them from file

        Returns
        -------
        dict: A dictionary containing all simulation_data

        """
        simulation_data = {}
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
            for experiment_data in experiment.experiment_attributes:
                if experiment_data.name.startswith('simulation_data_'):
                    simulation_data_name = experiment_data.name.split('simulation_data_', 1)[1]
                    simulation_data_name = simulation_data_name.split("_")
                    no_list = False
                    if simulation_data_name[-1] == "nolist":  # check if list or single str/float
                        no_list = True
                    simulation_data_name = "_".join(simulation_data_name[:-1])
                    # everything after, max 1 split
                    group_values = simulation_data.get(simulation_data_name, [])
                    if experiment_data.value is not None:
                        if no_list:
                            group_values = experiment_data.value
                        else:
                            group_values.append(experiment_data.value)
                    else:
                        if no_list:
                            group_values = experiment_data.str_value
                        else:
                            group_values.append(experiment_data.str_value)
                    simulation_data[simulation_data_name] = group_values

        return simulation_data

    @simulation_data.setter
    def simulation_data(self, value: dict):
        """Update simulation data

        Try to load the data from self.simulation_data_file, update the internals and update the yaml file.

        Parameters
        ----------
        value: dict
            A dictionary containing the (new) simulation data

        Returns
        -------
        Updates the internal simulation_data

        """
        if value is None:
            return

        is_nested = False
        for entry in value.values():
            if isinstance(entry, dict):
                log.warning("Converting nested dict of simulation_data into json_normalized version!")
                is_nested = True
        if is_nested:
            value = pd.json_normalize(value).to_dict(orient='records')[0]
            log.debug(value)

        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
            log.warning("Converting all values to float or str")
            for group in value:
                if isinstance(value[group], list):
                    for idx, group_value in enumerate(value[group]):
                        entry = get_or_create(ses, ExperimentAttribute, name=f"simulation_data_{group}_{idx}",
                                              experiment=experiment)
                        # TODO consider using a dedicated relationship database instead of two keys in the name?!
                        if isinstance(group_value, str):
                            entry.str_value = group_value
                        else:
                            entry.value = group_value
                else:
                    entry = get_or_create(ses, ExperimentAttribute, name=f"simulation_data_{group}_nolist",
                                          experiment=experiment)
                    if isinstance(value[group], str):
                        entry.str_value = value[group]
                    else:
                        entry.value = value[group]

            ses.commit()
