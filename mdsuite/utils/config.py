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

from dataclasses import dataclass


@dataclass
class Config:
    """Collection of MDSuite configurations.

    Attributes
    ----------
    bokeh_sizing_mode: str
        The way bokeh scales plots.
        see bokeh / sizing_mode for more information
    jupyter : bool
            If true, jupyter is being used.
    memory_fraction: bool
            The portion of the available memory to be used.

    """

    jupyter: bool = False
    bokeh_sizing_mode: str = "stretch_both"
    memory_fraction: float = 0.5


config = Config()

try:
    config.jupyter = get_ipython().__class__.__name__ == "ZMQInteractiveShell"
except NameError:
    pass

if config.jupyter:
    # change default in Jupyter Notebooks
    config.bokeh_sizing_mode = None
