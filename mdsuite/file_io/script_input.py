"""MDSuite script input module."""

import typing

import mdsuite.file_io.file_read
from mdsuite.database.simulation_database import TrajectoryChunkData, TrajectoryMetadata


class ScriptInput(mdsuite.file_io.file_read.FileProcessor):
    """
    For testing purposes. Does not actually process files, instead uses data given
     on instantiation.
    """

    def __init__(
        self, data: TrajectoryChunkData, metadata: TrajectoryMetadata, name: str
    ):
        """
        Provide all the data needed for this class to act as a FileProcessor

        Parameters
        ----------
        data
        metadata
        name : A unique name for this dataset. Used to prevent multiple adding of the
         same data.

        """
        self.data = data
        self.mdata = metadata
        self.name = name

        mdsuite.file_io.file_read.assert_species_list_consistent(
            data.species_list, metadata.species_list
        )
        if self.data.chunk_size != self.metadata.n_configurations:
            raise ValueError("Data must be provided in one chunk")

    def __str__(self):
        return self.name

    def _get_metadata(self) -> TrajectoryMetadata:
        return self.mdata

    def get_configurations_generator(
        self,
    ) -> typing.Iterator[mdsuite.file_io.file_read.TrajectoryChunkData]:
        yield self.data
