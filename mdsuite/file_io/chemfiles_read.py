"""Test MDSuites chemfiles read module."""

import pathlib
import typing

import chemfiles
import numpy as np
import tqdm

import mdsuite.database.mdsuite_properties
import mdsuite.file_io.file_read
import mdsuite.utils.meta_functions
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.database.simulation_database import TrajectoryMetadata


class ChemfilesRead(mdsuite.file_io.file_read.FileProcessor):
    """
    Read trajectory files via chemfiles.
    See https://chemfiles.org/chemfiles/0.10.0/formats.html for supported data formats.
    """

    def __init__(
        self,
        traj_file_path: typing.Union[str, pathlib.Path],
        topol_file_path: typing.Union[str, pathlib.Path] = None,
    ):
        """

        Parameters
        ----------
        traj_file_path
            Path to the trajectory file you want to read.
        topol_file_path : optional
            If the trajectory file does not contain all information about the topology of
            the system (i.e. which data in the trajectory file belongs to which particle),
             you can provide the topology here.

        """
        self.traj_file_path = pathlib.Path(traj_file_path).resolve()

        if topol_file_path is not None:
            topol_file_path = pathlib.Path(topol_file_path).resolve()
        self.topol_file_path = topol_file_path

        # until now, chemfiles only supports these 2 properties.
        # If more are added, link here the mdsuite property to the name of the property
        # attribute in chemfiles.Frame
        self.properties_to_chemfile_attr_dict = {
            # mdsuite_properties.unwrapped_positions: "positions",
            mdsuite_properties.positions: "positions",
            mdsuite_properties.velocities: "velocities",
        }

        # not all properties are guaranteed to be in the given file.
        # A subset of self.properties_to_chemfile_attr_dict will be extracted during
        # self._get_metadata
        self.properties_in_file = None
        self.species_name_to_line_idxs_dict = None
        self.batch_size = None

    def __str__(self):
        return str(self.traj_file_path)

    def _get_metadata(self) -> TrajectoryMetadata:
        """Get the necessary metadata out of chemfiles.

        Trajectory and the first chemfiles.Frame
        """
        with chemfiles.Trajectory(str(self.traj_file_path)) as traj:
            if self.topol_file_path is not None:
                traj.set_topology(str(self.topol_file_path))

            n_configs = traj.nsteps
            frame = traj.read()

        # get the box lengths from the first frame
        box_l = frame.cell.lengths

        # extract which lines in chemfiles.Frame.<property> belong to which species
        # by going through the atoms list
        self.species_name_to_line_idxs_dict = {}
        for line_idx, atm in enumerate(frame.atoms):
            name = atm.name
            if name not in self.species_name_to_line_idxs_dict.keys():
                self.species_name_to_line_idxs_dict[name] = []
            self.species_name_to_line_idxs_dict[name].append(line_idx)

        # see which properties are in the file. there is no way of finding out except
        # trying to access them
        self.properties_in_file = {}
        for (
            mds_prop,
            chemfile_attr_name,
        ) in self.properties_to_chemfile_attr_dict.items():
            try:
                frame.__getattribute__(chemfile_attr_name)
            except chemfiles.ChemfilesError:
                pass
            else:
                self.properties_in_file[mds_prop] = chemfile_attr_name

        species_list = []
        for key, val in self.species_name_to_line_idxs_dict.items():
            species_list.append(
                mdsuite.database.simulation_database.SpeciesInfo(
                    name=key,
                    n_particles=len(val),
                    properties=list(self.properties_in_file.keys()),
                )
            )

        return TrajectoryMetadata(
            n_configurations=n_configs,
            species_list=species_list,
            box_l=box_l,
        )

    def get_configurations_generator(
        self,
    ) -> typing.Iterator[mdsuite.file_io.file_read.TrajectoryChunkData]:
        """Implement parent abstract method."""
        batch_size = mdsuite.utils.meta_functions.optimize_batch_size(
            filepath=self.traj_file_path,
            number_of_configurations=self.metadata.n_configurations,
        )
        n_batches, n_configs_remainder = divmod(
            int(self.metadata.n_configurations), int(batch_size)
        )

        with chemfiles.Trajectory(str(self.traj_file_path)) as traj:
            if self.topol_file_path is not None:
                traj.set_topology(str(self.topol_file_path))
            for _ in tqdm.tqdm(range(n_batches), ncols=70):
                yield self._read_process_n_configurations(traj, batch_size)
            if n_configs_remainder > 0:
                yield self._read_process_n_configurations(traj, n_configs_remainder)

    def _read_process_n_configurations(
        self, traj: chemfiles.Trajectory, n_configs: int
    ) -> mdsuite.database.simulation_database.TrajectoryChunkData:
        """
        Read n configurations and package them into a trajectory chunk of the
        right format.

        Parameters
        ----------
        traj : chemfiles.Trajectory
            An open chemfiles Trajectory
        n_configs : int
            Number of configurations to read in.

        """
        species_list = self.metadata.species_list
        chunk = mdsuite.database.simulation_database.TrajectoryChunkData(
            species_list, n_configs
        )

        for i in range(n_configs):
            frame = traj.read()
            # slice by species
            for sp_info in species_list:
                for mds_prop, chemfile_attrname in self.properties_in_file.items():
                    data = frame.__getattribute__(chemfile_attrname)
                    idxs = self.species_name_to_line_idxs_dict[sp_info.name]
                    write_data = data[idxs, :]
                    # add 'time' axis. we only have one configuration to write
                    write_data = write_data[np.newaxis, :, :]
                    chunk.add_data(write_data, i, sp_info.name, mds_prop.name)
        return chunk
