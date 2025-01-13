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

from typing import TYPE_CHECKING

from .base import TimeSeries

if TYPE_CHECKING:
    from mdsuite import Experiment


class Energies(TimeSeries):
    """Energies Time Series class."""

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)
        self.loaded_property = "PE"
        self.fig_labels = {"x": r"Timestep $t$", "y": r"Energies $E$"}
