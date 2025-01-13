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


class CoordinateUnwrapper(SingleSpeciesTrafo):
    """
    Unwrap coordinates by checking if particles moved from one side of the box to
    the other within one time step.
    """

    def __init__(self):
        super(CoordinateUnwrapper, self).__init__(
            input_properties=[
                mdsuite_properties.positions,
                mdsuite_properties.box_length,
            ],
            output_property=mdsuite_properties.unwrapped_positions,
            scale_function={"linear": {"scale_factor": 2}},
        )

    def transform_batch(
        self, batch: typing.Dict[str, tf.Tensor], carryover: typing.Any = None
    ) -> typing.Tuple[tf.Tensor, dict]:
        """Implement parent class abstract method."""
        pos = batch[mdsuite_properties.positions.name]
        box_l = batch[mdsuite_properties.box_length.name]

        if carryover is None:
            last_pos = pos[:, 0, :]
            last_image_box = tf.zeros_like(last_pos)
        else:
            last_pos = carryover["last_pos"]
            last_image_box = carryover["last_image_box"]

        # calculate image box (write all in one variable to allow memory reusing)
        # calculate where jump happened
        image_box = tf.concat([tf.expand_dims(last_pos, axis=1), pos], axis=1)
        image_box = tf.experimental.numpy.diff(image_box, axis=1)
        image_box = tf.math.round(image_box / box_l)

        # sum up the jumps (negative bcs we need to go against
        # the jump that teleported the particle)
        image_box = -tf.math.cumsum(image_box, axis=1)

        # add past jumps (image_boxes)
        image_box += tf.expand_dims(last_image_box, axis=1)

        unwrapped_pos = pos + image_box * box_l

        carry = {"last_pos": pos[:, -1, :], "last_image_box": image_box[:, -1, :]}
        return unwrapped_pos, carry
