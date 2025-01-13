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
Module implementing the ZnVis visualizer in MDSuite.
"""

import importlib.resources
import json

import numpy as np
import znvis
from PIL.ImageColor import getcolor

import mdsuite.data
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.database.simulation_database import Database
from mdsuite.utils.meta_functions import join_path


class SimulationVisualizer:
    """Visualize a simulation."""

    def __init__(
        self,
        species: list = None,
        unwrapped: bool = False,
        frame_rate: float = 24,
        database_path: str = None,
    ):
        """
        Constructor for the visualizer.

        Parameters
        ----------
        species : list
                A list of species to visualize.
        unwrapped : bool
                If true, unwrapped coordinates are studied.
        frame_rate : float
                Frame rate at which to run the visualization in frames per second.
        database_path : str
                Database path from the experiment.

        """
        self.counter = 0
        # Particle information
        self.database = Database(join_path(database_path, "database.hdf5"))
        self.frame_rate = frame_rate
        self.species = species
        if unwrapped:
            self.identifier = mdsuite_properties.unwrapped_positions
        else:
            self.identifier = mdsuite_properties.positions

    @staticmethod
    def _get_species_properties(species: str):
        """
        Collect species properties from pubchempy file.

        Parameters
        ----------
        species : float
                Species to load data for.

        Returns
        -------
        colour : np.ndarray
                RBG array of colours.
        radius : float
                Radius of the particles. This is a reduced mass.

        """
        # Load the species data from pubchempy data file.
        pse = json.loads(
            importlib.resources.read_text(mdsuite.data, "PubChemElements_all.json")
        )

        colour = np.random.uniform(0, 1, size=(3,))
        radius = 1.0
        for entry in pse:
            if pse[entry][1] == species:
                colour = np.array(getcolor(f"#{pse[entry][4]}", "RGB")) / 255
                radius = float(pse[entry][3]) / 25

        return colour, radius

    def _prepare_species(self):
        """
        Build the znvis Particle objects.

        Returns
        -------
        particle_list : list[znvis.Particle]
                A list of particle objects.

        """
        particle_list = []
        for item in self.species:
            colour, radius = self._get_species_properties(item)
            trajectory = self.database.load_data(
                [join_path(item, self.identifier.name)], select_slice=np.s_[:]
            )
            trajectory = trajectory[join_path(item, self.identifier.name)]
            trajectory = np.transpose(trajectory, axes=[1, 0, 2])
            sphere = znvis.Sphere(colour=colour, radius=radius, resolution=10)
            particle_list.append(
                znvis.Particle(name=item, mesh=sphere, position=trajectory)
            )

        return particle_list

    def run_visualization(self):
        """
        Run the visualization.

        Returns
        -------
        Opens the ZnVis app and runs the visualization.

        """
        particle_list = self._prepare_species()
        visualizer = znvis.Visualizer(particles=particle_list, frame_rate=24)
        visualizer.run_visualization()
