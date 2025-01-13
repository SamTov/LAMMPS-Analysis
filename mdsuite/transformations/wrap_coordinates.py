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

import typing

import tensorflow as tf

from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.transformations.transformations import SingleSpeciesTrafo


class CoordinateWrapper(SingleSpeciesTrafo):
    """Wrap coordinates into the simulation box."""

    def __init__(self, center_box: bool = True):
        """
        Class init.

        Parameters
        ----------
        center_box: bool
            if True (default): coordinates are wrapped to [-L/2 , L/2]
            if False: coordinates are wrapped to [0 , L],
            where L is the box size.

        """
        super(CoordinateWrapper, self).__init__(
            input_properties=[
                mdsuite_properties.unwrapped_positions,
                mdsuite_properties.box_length,
            ],
            output_property=mdsuite_properties.positions,
            scale_function={"linear": {"scale_factor": 2}},
        )

        self.center_box: bool = center_box

    def transform_batch(
        self, batch: typing.Dict[str, tf.Tensor], carryover: typing.Any = None
    ) -> tf.Tensor:
        """Implement parent class abstract method."""
        unwrap_pos = batch[mdsuite_properties.unwrapped_positions.name]
        box_l = batch[mdsuite_properties.box_length.name]

        if self.center_box:
            unwrap_pos = unwrap_pos + box_l / 2.0
        pos = unwrap_pos - tf.floor(unwrap_pos / box_l) * box_l
        if self.center_box:
            pos = pos - box_l / 2.0

        return pos
