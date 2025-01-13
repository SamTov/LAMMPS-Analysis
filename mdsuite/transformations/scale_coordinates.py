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


class ScaleCoordinates(SingleSpeciesTrafo):
    """Scale coordinates by multiplying them with the box size."""

    def __init__(self):
        super(ScaleCoordinates, self).__init__(
            input_properties=[
                mdsuite_properties.scaled_positions,
                mdsuite_properties.box_length,
            ],
            output_property=mdsuite_properties.positions,
            scale_function={"linear": {"scale_factor": 2}},
        )

    def transform_batch(
        self, batch: typing.Dict[str, tf.Tensor], carryover: typing.Any = None
    ) -> tf.Tensor:
        """Implement parent class abstract method."""
        pos = batch[mdsuite_properties.scaled_positions.name]
        box_l = batch[mdsuite_properties.box_length.name]
        return pos * box_l
