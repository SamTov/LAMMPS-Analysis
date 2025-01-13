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

import pathlib
from typing import List, Union

import numpy as np
from bokeh.io import output_file, output_notebook
from bokeh.layouts import gridplot
from bokeh.models import HoverTool
from bokeh.plotting import figure, show

from mdsuite.utils import config


class DataVisualizer2D:
    """Visualizer for two-dimensional data."""

    def __init__(self, title: str, path: pathlib.Path):
        """
        Constructor for the data visualizer.

        Parameters
        ----------
        title : str
                title of the plot.
        path : pathlib.Path
                path to the saving directory of the plot

        """
        if config.jupyter:
            output_notebook()
        else:
            output_file(f"{path / title}.html")

    def construct_plot(
        self,
        x_data: Union[list, np.ndarray],
        y_data: Union[list, np.ndarray],
        x_label: str,
        y_label: str,
        title: str,
        layouts: List = None,
    ) -> figure:
        """
        Generate a plot.

        Parameters
        ----------
        x_data : Union[list, np.ndarray, tf.Tensor]
                data to plot along the x axis.
        y_data : Union[list, np.ndarray, tf.Tensor]
                data to plot along the y axis.
        x_label : str
                label for the x axis
        y_label : str
                label of the y axis.
        title : str
                name of the specific plot.

        Returns
        -------
        figure : figure
                A bokeh figure object.

        """
        fig = figure(
            x_axis_label=x_label,
            y_axis_label=y_label,
            sizing_mode=config.bokeh_sizing_mode,
        )
        fig.line(x_data, y_data, legend_label=title)
        fig.add_tools(HoverTool())
        if layouts is not None:
            for item in layouts:
                fig.add_layout(item)

        return fig

    def grid_show(self, figures: list):
        """
        Display a list of figures in a grid.

        Parameters
        ----------
        figures : list
                A list of figures to display.

        Returns
        -------

        """
        grid = gridplot(figures, ncols=3)
        show(grid)
