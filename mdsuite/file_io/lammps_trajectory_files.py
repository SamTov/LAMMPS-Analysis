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
"""

import pathlib
import typing

import numpy as np

import mdsuite.file_io.tabular_text_files
import mdsuite.utils.meta_functions
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.database.simulation_database import TrajectoryMetadata
from mdsuite.file_io.tabular_text_files import (
    get_species_list_from_tabular_text_reader_data,
)
from mdsuite.utils.meta_functions import sort_array_by_column

column_names = {
    mdsuite_properties.positions: ["x", "y", "z"],
    mdsuite_properties.scaled_positions: ["xs", "ys", "zs"],
    mdsuite_properties.unwrapped_positions: ["xu", "yu", "zu"],
    mdsuite_properties.scaled_unwrapped_positions: ["xsu", "ysu", "zsu"],
    mdsuite_properties.velocities: ["vx", "vy", "vz"],
    mdsuite_properties.forces: ["fx", "fy", "fz"],
    mdsuite_properties.box_images: ["ix", "iy", "iz"],
    mdsuite_properties.dipole_orientation_magnitude: ["mux", "muy", "muz"],
    mdsuite_properties.angular_velocity_spherical: ["omegax", "omegay", "omegaz"],
    mdsuite_properties.angular_velocity_non_spherical: [
        "angmomx",
        "angmomy",
        "angmomz",
    ],
    mdsuite_properties.torque: ["tqx", "tqy", "tqz"],
    mdsuite_properties.charge: ["q"],
    mdsuite_properties.kinetic_energy: ["c_KE"],
    mdsuite_properties.potential_energy: ["c_PE"],
    mdsuite_properties.stress: [
        "c_Stress[1]",
        "c_Stress[2]",
        "c_Stress[3]",
        "c_Stress[4]",
        "c_Stress[5]",
        "c_Stress[6]",
    ],
}


class LAMMPSTrajectoryFile(mdsuite.file_io.tabular_text_files.TabularTextFileProcessor):
    """Reader for LAMMPS files."""

    def __init__(
        self,
        file_path: typing.Union[str, pathlib.Path],
        trajectory_is_sorted_by_ids=False,
        custom_data_map: dict = None,
    ):
        """

        Parameters
        ----------
        file_path:
            path to the LAMMPS trajectory file
        trajectory_is_sorted_by_ids:
            Flag to indicate if the particle order in the trajectory file is the same for
            each configuration
        custom_data_map
            If your file contains columns with data that is not part of the standard set
            of properties (see column_names in this file),
            you can map the column names to the corresponding property.
            example: custom_data_map = {"Reduced_Momentum": ["rp_x", "rp_y", "rp_z"]},
            if the file contains columns labelled as 'rp_{x,y,z}' for the three components
            of the reduced momentum vector

        """
        super(LAMMPSTrajectoryFile, self).__init__(
            file_path,
            file_format_column_names=column_names,
            custom_column_names=custom_data_map,
        )
        self.n_header_lines = 9
        self.trajectory_is_sorted_by_ids = trajectory_is_sorted_by_ids

    def _get_tabular_text_reader_mdata(
        self,
    ) -> mdsuite.file_io.tabular_text_files.TabularTextFileReaderMData:
        """Implement abstract parent method."""
        with open(self.file_path, "r") as file:
            header = mdsuite.file_io.tabular_text_files.read_n_lines(
                file, self.n_header_lines, start_at=0
            )

            # extract data that can be directly read off the header
            n_particles = int(header[3].split()[0])

            # extract properties from the column names
            header_property_names = header[8].split()[2:]
            id_column_idx = header_property_names.index("id")
            property_dict = extract_properties_from_header(
                header_property_names, self._column_name_dict
            )

            # get number of configs from file length
            file.seek(0)
            num_lines = sum(1 for _ in file)
            n_configs_float = num_lines / (n_particles + self.n_header_lines)
            n_configs = int(round(n_configs_float))
            assert abs(n_configs_float - n_configs) < 1e-10

            # get information on which particles with which id belong to which species
            # by analysing the first configuration
            file.seek(0)
            species_dict = self._get_species_information(
                file, header_property_names, n_particles
            )

        return mdsuite.file_io.tabular_text_files.TabularTextFileReaderMData(
            n_configs=n_configs,
            species_name_to_line_idx_dict=species_dict,
            property_to_column_idx_dict=property_dict,
            n_header_lines=self.n_header_lines,
            n_particles=n_particles,
            header_lines_for_each_config=True,
            sort_by_column_idx=(
                None if self.trajectory_is_sorted_by_ids else id_column_idx
            ),
        )

    def _get_metadata(self) -> TrajectoryMetadata:
        """
        Gets the metadata for database creation as an implementation of the parent class
        virtual function.
        """
        with open(self.file_path, "r") as file:
            header = mdsuite.file_io.tabular_text_files.read_n_lines(
                file, self.n_header_lines, start_at=0
            )
            header_boxl_lines = header[5:8]
            box_l = [
                float(line.split()[1]) - float(line.split()[0])
                for line in header_boxl_lines
            ]
            # extract sample step information from consecutive headers
            file.seek(0)
            sample_rate = self._get_sample_rate(
                file, self.tabular_text_reader_data.n_particles
            )

        species_list = get_species_list_from_tabular_text_reader_data(
            self.tabular_text_reader_data
        )

        mdata = TrajectoryMetadata(
            n_configurations=self.tabular_text_reader_data.n_configs,
            species_list=species_list,
            box_l=box_l,
            sample_rate=sample_rate,
        )

        return mdata

    def _get_species_information(
        self, file, header_property_names: list, n_particles: int
    ):
        """
        Get the information which species are present and which particle ids/ lines in
        the file belong to them.

        Parameters
        ----------
        file: file object
            the open file object to read from
        header_property_names: list
            list of the names of the columns in the file

        """
        header_id_index = header_property_names.index("id")
        #
        # Look for element keyword in trajectory.
        if "element" in header_property_names:
            header_species_index = header_property_names.index("element")
        # Look for type keyword if element is not present.
        elif "type" in header_property_names:
            header_species_index = header_property_names.index("type")
        # Raise an error if no identifying keywords are found.
        else:
            raise ValueError("Insufficient species or type identification available.")

        species_dict = {}
        # skip the header
        mdsuite.file_io.tabular_text_files.skip_n_lines(file, self.n_header_lines)
        # read one configuration
        traj_data = np.stack(
            [np.array(list(file.readline().split())) for _ in range(n_particles)]
        )
        # sort by particle id
        if not self.trajectory_is_sorted_by_ids:
            traj_data = sort_array_by_column(traj_data, header_id_index)
        # iterate over the first configuration, whenever a new species
        # (value at species_index) is encountered, add an entry
        for i, line in enumerate(traj_data):
            species_name = line[header_species_index]
            if species_name not in species_dict.keys():
                species_dict[species_name] = []
            species_dict[species_name].append(i)

        return species_dict

    def _get_sample_rate(self, file, n_particles: int) -> typing.Union[int, None]:
        first_header = mdsuite.file_io.tabular_text_files.read_n_lines(
            file, self.n_header_lines, start_at=0
        )
        time_step_0 = int(first_header[1])  # Time in first configuration
        # catch single snapshot trajectory (second_header == [])
        try:
            second_header = mdsuite.file_io.tabular_text_files.read_n_lines(
                file, self.n_header_lines, start_at=self.n_header_lines + n_particles
            )
        except StopIteration:
            return None

        time_step_1 = int(second_header[1])  # Time in second configuration
        return time_step_1 - time_step_0


def extract_properties_from_header(
    header_property_names: list, database_correspondence_dict: dict
) -> dict:
    """
    Takes the property names from a file header, sees if there is a corresponding
    mdsuite property in database_correspondence_dict.
    Returns a dict that links the mdsuite property names to the column indices at which
    they can be found in the file.

    Parameters
    ----------
    header_property_names
        The names of the columns in the data file
    database_correspondence_dict
        The translation between mdsuite properties and the column names of the
        respective file format.
        lammps example:
        {"Positions": ["x", "y", "z"],
         "Unwrapped_Positions": ["xu", "yu", "zu"],

    Returns
    -------
    trajectory_properties : dict
        A dict of the form
        {'MDSuite_Property_1': [column_indices], 'MDSuite_Property_2': ...}
        Example {'Unwrapped_Positions': [2,3,4], 'Velocities': [5,6,8]}

    """
    column_dict_properties = {
        variable: idx for idx, variable in enumerate(header_property_names)
    }
    # for each property label (position, velocity,etc) in the lammps definition
    for property_label, property_names in database_correspondence_dict.items():
        # for each coordinate for a given property label (position: x, y, z),
        # get idx and the name
        for idx, property_name in enumerate(property_names):
            if (
                property_name in column_dict_properties.keys()
            ):  # if this name (x) is in the input file properties
                # we change the lammps_properties_dict replacing the string of the
                # property name by the column name
                database_correspondence_dict[property_label][idx] = (
                    column_dict_properties[property_name]
                )

    # trajectory_properties only need the labels with the integer columns, then we
    # only copy those
    trajectory_properties = {}
    for property_label, properties_columns in database_correspondence_dict.items():
        if all(
            isinstance(property_column, int) for property_column in properties_columns
        ):
            trajectory_properties[property_label] = properties_columns

    return trajectory_properties
