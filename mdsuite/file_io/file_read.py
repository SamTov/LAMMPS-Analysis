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

from __future__ import annotations

import abc
import typing

from mdsuite.database.simulation_database import TrajectoryChunkData, TrajectoryMetadata


class FileProcessor(abc.ABC):
    """
    Class to handle reading from trajectory files.
    Output is supposed to be used by the experiment class for building and populating
    trajectory database.
    """

    _metadata: TrajectoryMetadata = None

    @abc.abstractmethod
    def __str__(self):
        """
        Return a unique string representing this FileProcessor. (
        The absolute file path, for example).
        """
        raise NotImplementedError("File Processors must implement a string")

    @abc.abstractmethod
    def _get_metadata(self) -> TrajectoryMetadata:
        """
        Return the metadata required to build a database.
        See mdsuite.database.simulation_database.TrajectoryMetadata to see the details of
        which values must be extracted from the file.
        """
        raise NotImplementedError("File Processors must implement metadata extraction")

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self._get_metadata()
        return self._metadata

    @abc.abstractmethod
    def get_configurations_generator(self) -> typing.Iterator[TrajectoryChunkData]:
        """
        Yield configurations as chunks. Batch size must be determined by the
        FileProcessor. See mdsuite.database.simulation_database.TrajectoryChunkData for
        the format in which to provide the trajectory chunk.

        Returns
        -------
        generator that yields TrajectoryChunkData

        """
        raise NotImplementedError("File Processors must implement data loading")


def assert_species_list_consistent(sp_list_0, sp_list_1):
    for sp_info_data, sp_info_mdata in zip(sp_list_0, sp_list_1):
        if sp_info_data != sp_info_mdata:
            raise ValueError("Species information in the two lists is inconsistent")
