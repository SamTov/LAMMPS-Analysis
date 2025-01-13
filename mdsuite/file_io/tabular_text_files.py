"""MDSuite Tabular Text file reader module."""

import abc
import copy
import dataclasses
import pathlib
import typing

import numpy as np
import tqdm

import mdsuite.database.simulation_database
import mdsuite.file_io.file_read
import mdsuite.utils.meta_functions


@dataclasses.dataclass
class TabularTextFileReaderMData:
    """Class to hold the data that needs to be extracted from tabular text files before
    reading them.

    Attributes
    ----------
    n_configs:
        Number of configs in the file
    n_particles:
        Total number of particles
    species_name_to_line_idx_dict:
        A dict that links the species name to the line idxs at which the particles can be
        found within a configuration.
        Example: {"Na":[0,2,4], "Cl":[1,3,5]} for a file in which Na and Cl are written
        alternatingly.
    property_to_column_idx_dict
        A dict that links the property name to the column idxs at which the property
        is listed. Usually the output of
        mdsuite.file_io.tabular_text_files.extract_properties_from_header
    n_header_lines:
        Number of header lines PER CONFIG
    header_lines_for_each_config:
        Flag to indicate wether each config has its own header or if there is just one
        header at the top of the file.
    sort_by_column_idx:
        if None (default): no sorting needed (the particles are always in the same order
        within a config
        if int: sort the lines in the config by the column with this index
        (e.g., use to sort by particle id in unsorted config output)

    """

    n_configs: int
    species_name_to_line_idx_dict: typing.Dict[str, list]
    n_particles: int
    property_to_column_idx_dict: typing.Dict[str, list]
    n_header_lines: int
    header_lines_for_each_config: bool = False
    sort_by_column_idx: int = None


class TabularTextFileProcessor(mdsuite.file_io.file_read.FileProcessor):
    """Parent class for all file readers that are based on tabular text data
    (e.g. lammps, extxyz,...).
    """

    def __init__(
        self,
        file_path: typing.Union[str, pathlib.Path],
        file_format_column_names: typing.Dict[
            mdsuite.database.simulation_database.PropertyInfo, list
        ] = None,
        custom_column_names: typing.Dict[str, typing.Any] = None,
    ):
        """
        Init, also handles the combination of file_format_column_names and
        custom_column_names.
        The result, self._column_name_dict is supposed to be used by child functions to
        create their TabularTextFileReaderData

        Parameters
        ----------
        file_path:
            Path to the tabular text file.
        file_format_column_names
            Dict connecting mdsuite properties (as defined in
            mdsuite.database.mdsuite_properties.mdsuite_properties) the columns of the
            file format. Constant to be provided by the child classes.
            Example: {mdsuite_properties.positions: ["x", "y", "z"]}
        custom_column_names:
            Dict connecting user-defined properties the column names. To be provided by
            the user.
            Example: {'MyMagicProperty':['MMP1', 'MMP2']}.

        """
        self.file_path = pathlib.Path(file_path).resolve()
        my_file_format_column_names = copy.deepcopy(file_format_column_names)
        if my_file_format_column_names is None:
            my_file_format_column_names = {}
        str_file_format_column_names = {
            prop.name: val for prop, val in my_file_format_column_names.items()
        }

        if custom_column_names is None:
            custom_column_names = {}
        str_file_format_column_names.update(custom_column_names)
        self._column_name_dict = str_file_format_column_names

        self._tabular_text_reader_mdata: TabularTextFileReaderMData = None

    @abc.abstractmethod
    def _get_tabular_text_reader_mdata(self) -> TabularTextFileReaderMData:
        """
        Child classes of TabularTextFileProcessor must implement this function, so its
        output can be used in get_configurations_generator.
        See TabularTextFileReaderData for the data that needs to be provided.
        """
        raise NotImplementedError("Tabular text files must implement this function")

    @property
    def tabular_text_reader_data(self) -> TabularTextFileReaderMData:
        if self._tabular_text_reader_mdata is None:
            self._tabular_text_reader_mdata = self._get_tabular_text_reader_mdata()
        return self._tabular_text_reader_mdata

    def __str__(self):
        return str(self.file_path)

    def get_configurations_generator(
        self,
    ) -> typing.Iterator[mdsuite.database.simulation_database.TrajectoryChunkData]:
        """
        TabularTextFiles implements the parent virtual function,
        but requires its children to provide the necessary information about the table
        contents,
        see self._get_tabular_text_reader_data.
        """
        n_configs = self.tabular_text_reader_data.n_configs

        batch_size = mdsuite.utils.meta_functions.optimize_batch_size(
            filepath=self.file_path, number_of_configurations=n_configs
        )
        n_batches, n_configs_remainder = divmod(int(n_configs), int(batch_size))

        with open(self.file_path, "r") as file:
            file.seek(0)
            # skip header either once in the beginning or for each config
            if self.tabular_text_reader_data.header_lines_for_each_config:
                n_header_lines_in_config = self.tabular_text_reader_data.n_header_lines
            else:
                skip_n_lines(file, self.tabular_text_reader_data.n_header_lines)
                n_header_lines_in_config = 0

            for _ in tqdm.tqdm(range(n_batches), ncols=70):
                yield self._read_process_n_configurations(
                    file,
                    batch_size,
                    n_header_lines=n_header_lines_in_config,
                )
            if n_configs_remainder > 0:
                yield self._read_process_n_configurations(
                    file,
                    n_configs_remainder,
                    n_header_lines=n_header_lines_in_config,
                )

    def _read_process_n_configurations(
        self,
        file,
        n_configs: int,
        n_header_lines: int = 0,
    ) -> mdsuite.database.simulation_database.TrajectoryChunkData:
        """
        Read n configurations and package them into a trajectory chunk of the right format

        Parameters
        ----------
        file:
            A file opened at the start of a configuration
        n_configs:
            Number of configs to process
        n_header_lines:
            Number of header lines PER CONFIG
        -------
            The chunk for your reader output.

        """
        species_list = self.metadata.species_list
        chunk = mdsuite.database.simulation_database.TrajectoryChunkData(
            species_list, n_configs
        )

        for config_idx in range(n_configs):
            # skip the header
            mdsuite.file_io.tabular_text_files.skip_n_lines(file, n_header_lines)
            # read one config
            traj_data = np.stack(
                [
                    (list(file.readline().split()))
                    for _ in range(
                        self.tabular_text_reader_data.n_particles,
                    )
                ]
            )
            # sort by id
            if self.tabular_text_reader_data.sort_by_column_idx is not None:
                traj_data = mdsuite.utils.meta_functions.sort_array_by_column(
                    traj_data, self.tabular_text_reader_data.sort_by_column_idx
                )

            # slice by species
            for sp_info in species_list:
                idxs = self.tabular_text_reader_data.species_name_to_line_idx_dict[
                    sp_info.name
                ]
                sp_data = traj_data[idxs, :]
                # slice by property
                for prop_info in sp_info.properties:
                    prop_column_idxs = (
                        self.tabular_text_reader_data.property_to_column_idx_dict[
                            prop_info.name
                        ]
                    )
                    write_data = sp_data[:, prop_column_idxs]
                    # add 'time' axis. we only have one configuration to write
                    write_data = write_data[np.newaxis, :, :]
                    chunk.add_data(write_data, config_idx, sp_info.name, prop_info.name)

        return chunk


def read_n_lines(file, n_lines: int, start_at: int = None) -> list:
    """
    Get n_lines lines, starting at line number start_at.
    If start_at is None, read from the current file state

    Returns
    -------
    A list of strings, one string for each line.

    """
    if start_at is not None:
        file.seek(0)
        skip_n_lines(file, start_at)
    return [next(file) for _ in range(n_lines)]


def skip_n_lines(file, n_lines: int) -> None:
    """
    Skip n_lines in file

    Parameters
    ----------
    file: the file where we skip lines
    n_lines: the number of lines to skip.

    Returns
    -------
        Nothing

    """
    for _ in range(n_lines):
        next(file)


def get_species_list_from_tabular_text_reader_data(
    tabular_text_reader_data: TabularTextFileReaderMData,
) -> typing.List[mdsuite.database.simulation_database.SpeciesInfo]:
    """
    Use the data collected in TabularTextFileProcessor._get_tabular_text_reader_data() to
    get the values necessary for
    TabularTextFileProcessor.metadata.
    """
    # all species have the same properties
    properties_list = []
    for (
        key,
        val,
    ) in tabular_text_reader_data.property_to_column_idx_dict.items():
        properties_list.append(
            mdsuite.database.simulation_database.PropertyInfo(name=key, n_dims=len(val))
        )

    species_list = []
    for key, val in tabular_text_reader_data.species_name_to_line_idx_dict.items():
        species_list.append(
            mdsuite.database.simulation_database.SpeciesInfo(
                name=key,
                n_particles=len(val),
                properties=properties_list,
            )
        )

    return species_list
