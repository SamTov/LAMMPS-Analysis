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
import logging
import pathlib
import typing

import numpy as np

import mdsuite.database.simulation_database
import mdsuite.file_io.file_read
import mdsuite.file_io.tabular_text_files
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.file_io.tabular_text_files import (
    get_species_list_from_tabular_text_reader_data,
)

log = logging.getLogger(__name__)

var_names = {
    mdsuite_properties.positions: "pos",
    mdsuite_properties.velocities: "vel",
    mdsuite_properties.forces: "force",
    mdsuite_properties.stress: "stress",
    mdsuite_properties.energy: "energies",
    mdsuite_properties.time: "time",
    mdsuite_properties.momenta: "momenta",
}


class EXTXYZFile(mdsuite.file_io.tabular_text_files.TabularTextFileProcessor):
    """Reader for extxyz files."""

    def __init__(
        self, file_path: typing.Union[str, pathlib.Path], custom_data_map: dict = None
    ):
        """

        Parameters
        ----------
        file_path
            Path to the extxyz file
        custom_data_map:
            If your file contains columns with data that is not part of the standard
            set of properties (see var_names in this file),
            you can map the column names to the corresponding property.
            example: custom_data_map = {"Reduced_Momentum": "redmom"},
            if the file header contains "redmom:R:3" to point to the correct 3
            columns containing the reduced momentum values

        """
        super(EXTXYZFile, self).__init__(
            file_path,
            file_format_column_names=var_names,
            custom_column_names=custom_data_map,
        )
        self.n_header_lines = 2

    def _get_tabular_text_reader_mdata(
        self,
    ) -> mdsuite.file_io.tabular_text_files.TabularTextFileReaderMData:
        """Implement abstract parent method."""
        with open(self.file_path, "r") as file:
            # first header line: number of particles
            n_particles = int(file.readline())
            # second line: other info
            header = file.readline()

            species_idx, property_dict = _get_property_to_column_idx_dict(
                header, self._column_name_dict
            )

            file.seek(0)
            species_dict = self._get_species_information(file, species_idx, n_particles)

            # get number of configs from file length
            file.seek(0)
            num_lines = sum(1 for _ in file)
            n_configs_float = num_lines / (n_particles + self.n_header_lines)
            n_configs = int(round(n_configs_float))
            assert abs(n_configs_float - n_configs) < 1e-10

            return mdsuite.file_io.tabular_text_files.TabularTextFileReaderMData(
                n_configs=n_configs,
                species_name_to_line_idx_dict=species_dict,
                property_to_column_idx_dict=property_dict,
                n_header_lines=self.n_header_lines,
                n_particles=n_particles,
                header_lines_for_each_config=True,
            )

    def _get_metadata(self):
        """
        Gets the metadata for database creation as an implementation of the parent class
        virtual function by analysing the header lines and one full configuration.
        """
        with open(self.file_path, "r") as file:
            file.readline()
            # box_l in second header line
            header = file.readline()
            box_l = _get_box_l(header)

            file.seek(0)
            mdsuite.file_io.tabular_text_files.skip_n_lines(
                file,
                self.tabular_text_reader_data.n_particles + self.n_header_lines + 1,
            )
            header_1 = file.readline()

            time_0 = _get_time(header)
            time_1 = _get_time(header_1)
            if time_0 is not None and time_1 is not None:
                sample_rate = int(round(time_1 - time_0))
            else:
                log.warning(
                    "Could not read sample rate from file. Please adjust the sample rate"
                    " manually if required."
                )
                sample_rate = None

        species_list = get_species_list_from_tabular_text_reader_data(
            self.tabular_text_reader_data
        )

        mdata = mdsuite.database.simulation_database.TrajectoryMetadata(
            n_configurations=self.tabular_text_reader_data.n_configs,
            box_l=box_l,
            sample_rate=sample_rate,
            species_list=species_list,
        )

        return mdata

    def _get_species_information(self, file, species_idx: int, n_particles: int):
        """
        Get the initial species information.

        Parameters
        ----------
        file:
            An opened extxyz file
        species_idx:
            The index of the column in which the species name is stored
        n_particles:
            The total number of particles

        """
        mdsuite.file_io.tabular_text_files.skip_n_lines(file, self.n_header_lines)
        # read one configuration
        traj_data = np.stack([list(file.readline().split()) for _ in range(n_particles)])

        # Loop over atoms in first configuration.
        species_dict = {}
        for i, line in enumerate(traj_data):
            sp_name = line[species_idx]
            if sp_name not in list(species_dict.keys()):
                species_dict[sp_name] = []
            species_dict[sp_name].append(i)

        return species_dict


def _get_box_l(header: str) -> list:
    """
    Get the box lengths from the Lattice property in the header.

    Parameters
    ----------
    header:
        The extxyz header line as one string

    Returns
    -------
    [box_l_x, box_l_y, box_l_z]
        The tree sides of the box cuboid

    """
    data = copy.deepcopy(header).split()
    lattice = None
    start = None
    stop = None
    for idx, item in enumerate(data):
        if "Lattice" in item:
            start = idx
            break

    if start is not None:
        for idx, item in enumerate(data[start:]):
            if item[-1] == '"':
                stop = idx
                break
    else:
        raise RuntimeError("Could not find lattice size in file header")

    if stop is not None:
        lattice = data[start : stop + 1]
        lattice[0] = lattice[0].split("=")[1].replace('"', "")
        lattice[-1] = lattice[-1].replace('"', "")
        lattice = np.array(lattice).astype(float)

    return [lattice[0], lattice[4], lattice[8]]


def _get_time(header: str) -> float:
    """
    Retrieve the time value from a header line.
    Can be used to infer the sampling step by calling on consecutive config headers.

    Parameters
    ----------
    header
        The extxyz header line as one string.

    """
    data = copy.deepcopy(header).split()
    time = None
    for item in data:
        if var_names[mdsuite_properties.time] in item:
            try:
                time = float(item.split("=")[-1])
            except ValueError:
                time = float(item.split("=")[-1].split(",")[0])
    return time


def _get_property_to_column_idx_dict(
    header: str, var_names: dict
) -> typing.Tuple[int, typing.Dict[str, typing.List[int]]]:
    """
    Get the property summary from the header data.

    Parameters
    ----------
    header:
        header to analyse
    var_names:
        dict of translations from MDsuite property names to extxyz property names

    Returns
    -------
    species_index: int
        The index of the column in which the species names are stored
    property_summary : dict
            A dictionary of properties and their location in the data file.

    """
    data = copy.deepcopy(header).split()
    properties_string = None
    for item in data:
        if "Properties" in item:
            properties_string = item
    if properties_string is None:
        raise RuntimeError("Could not find properties in header")

    properties_list = (
        properties_string.split("=")[1].replace(":S", "").replace(":R", "").split(":")
    )
    property_summary = {}
    species_index = None
    index = 0
    var_names_values = list(var_names.values())
    var_names_keys = list(var_names.keys())
    for idx, item in enumerate(properties_list):
        if item == "species":
            species_index = index
            index += 1
        if item in var_names_values:
            key = var_names_keys[var_names_values.index(item)]
            length = int(properties_list[idx + 1])
            property_summary[key] = [index + i for i in range(length)]
            index += length

    if species_index is None:
        raise RuntimeError("could not find species in header")

    return species_index, property_summary
