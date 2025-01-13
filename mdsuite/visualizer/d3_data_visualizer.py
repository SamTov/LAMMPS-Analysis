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
Module for the MDSuite 3d visualizer.
"""

import importlib.resources
import json
from typing import Union

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from matplotlib import cm
from PIL.ImageColor import getcolor


class DataVisualizer3D:
    """Class for the visualizer of three dimensional data."""

    def __init__(
        self,
        data: np.ndarray,
        title: str,
        center: Union[str, dict] = None,
        colour_map: np.ndarray = None,
    ):
        """
        Constructor for the data visualizer.

        Parameters
        ----------
        data : np.ndarray
                data to plot.
        center : str
                centre molecule to be displayed (optional)
        title : str
                title of the plot.
        colour_map : np.ndarray
                A colour map to apply to the data.

        """
        self.data = data
        self.title = title
        self.center = center
        self.colour_map = colour_map
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(self.data)
        self.point_cloud.colors = o3d.utility.Vector3dVector(self._build_colour_map())

        self._build_app()

    def _build_colour_map(self):
        """
        Build the colour map
        Returns
        -------.

        """
        viridis = cm.get_cmap("binary")

        dat = viridis(self.colour_map)[:, :-1]

        return dat

    def _build_app(self):
        """
        Build the app window and update the class.

        Returns
        -------
        Updates the class.

        """
        self.app = gui.Application.instance
        self.app.initialize()

        self.vis = o3d.visualization.O3DVisualizer("MDSuite Visualizer", 1024, 768)
        self.vis.show_settings = True
        self.vis.reset_camera_to_default()

        self.app.add_window(self.vis)

    @staticmethod
    def _get_atom_properties(element: str) -> dict:
        """
        Get atom size and colour based on species.

        Parameters
        ----------
        element : str
                Name of the element you want to render.

        Returns
        -------
        data : dict
                 A dictionary of data to use for the rendering:
                 e.g. {'colour': (0.7, 0.33, 0.0), 'mass': 0.8)

        """
        data = {}
        data_name = "mdsuite.data"
        json_name = "PubChemElements_all.json"
        with importlib.resources.open_text(data_name, json_name) as json_file:
            pse = json.loads(json_file.read())

        try:
            name_split = element.split("_")
            element = name_split[0]
        except ValueError:
            element = element

        for entry in pse:
            if pse[entry][1] == element:
                data["colour"] = getcolor(f"#{pse[entry][4]}", "RGB")
                data["mass"] = float(pse[entry][3])

        return data

    def _add_center(self):
        """
        Add a rendering to the (0, 0, 0) point in the plot.

        Returns
        -------
        Updates the plot.

        """
        if type(self.center) is str:
            self._add_single_center()
        else:
            self._add_group_center()

    def _add_single_center(self):
        """
        Add a single particle to the center.

        Returns
        -------

        """
        data = self._get_atom_properties(self.center)
        mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius=data["mass"] / 25, resolution=15
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(data["colour"])
        self.vis.add_geometry("Center", mesh)

    def _add_group_center(self):
        """
        Add a group of particle to the center.

        Returns
        -------
        Adds a group of particles to the center.

        """
        translation = np.array([0, 0, 0])
        mass = 0.0
        for item in self.center:
            data = self._get_atom_properties(item)
            mass += data["mass"]
            translation += (self.center[item] / 10) * data["mass"]
        translation = -1 * translation / mass

        global_mesh = None
        for i, item in enumerate(self.center):
            data = self._get_atom_properties(item)
            if i == 0:
                global_mesh = o3d.geometry.TriangleMesh.create_sphere(
                    radius=data["mass"] / 100, resolution=15
                )
                global_mesh.compute_vertex_normals()
                global_mesh.translate(self.center[item] / 10)
                colour = np.array(data["colour"]) / 255
                global_mesh.paint_uniform_color(colour)
                # self.vis.add_geometry(item, global_mesh)
            else:
                mesh = o3d.geometry.TriangleMesh.create_sphere(
                    radius=data["mass"] / 100, resolution=15
                )
                mesh.compute_vertex_normals()
                mesh.translate(self.center[item] / 10)
                colour = np.array(data["colour"]) / 255
                mesh.paint_uniform_color(colour)
                global_mesh += mesh
                # self.vis.add_geometry(item, mesh)
        global_mesh.translate(translation)
        self.vis.add_geometry("Center", global_mesh)

    def plot(self):
        """
        Plot the data.

        Returns
        -------

        """
        self.vis.add_geometry("Points", self.point_cloud)

        # if self.center is not None:
        self._add_center()

        self.vis.reset_camera_to_default()
        self.app.run()
