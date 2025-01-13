"""Velocity from positions Trafo module.

This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html.

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Summary
-------
Calculate the velocity of particles from their positions
"""

import typing

import tensorflow as tf

from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.transformations.transformations import SingleSpeciesTrafo


class VelocityFromPositions(SingleSpeciesTrafo):
    """
    Calculate the velocity based on the particle positions via simple forward derivative,
    i.e. v(t) = (x(t+dt)-x(t))/dt.
    The last velocity of the trajectory cannot be computed and is copied
    from the second to last.
    """

    def __init__(self):
        super(VelocityFromPositions, self).__init__(
            input_properties=[
                mdsuite_properties.unwrapped_positions,
                mdsuite_properties.time_step,
                mdsuite_properties.sample_rate,
            ],
            output_property=mdsuite_properties.velocities_from_positions,
            scale_function={"linear": {"scale_factor": 2}},
        )

    def transform_batch(
        self, batch: typing.Dict[str, tf.Tensor], carryover: typing.Any = None
    ) -> tf.Tensor:
        """Implement parent class abstract method."""
        pos = batch[mdsuite_properties.unwrapped_positions.name]
        dt = (
            batch[mdsuite_properties.time_step.name]
            * batch[mdsuite_properties.sample_rate.name]
        )

        pos_plus_dt = tf.roll(pos, shift=-1, axis=1)
        vel = (pos_plus_dt - pos) / dt
        # discard last value, it comes from wrapping around the positions
        vel = vel[:, :-1, :]
        # instead, append the last value again
        last_values = vel[:, -1, :]
        return tf.concat((vel, last_values[:, None, :]), axis=1)
