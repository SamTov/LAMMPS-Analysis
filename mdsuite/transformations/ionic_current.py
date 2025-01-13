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
from mdsuite.transformations.transformations import MultiSpeciesTrafo


class IonicCurrent(MultiSpeciesTrafo):
    """Transformation to calculate the ionic current (charge * velocities)."""

    def __init__(self):
        super(IonicCurrent, self).__init__(
            input_properties=[
                mdsuite_properties.velocities,
                mdsuite_properties.charge,
            ],
            output_property=mdsuite_properties.ionic_current,
            scale_function={"linear": {"scale_factor": 2}},
        )

    def transform_batch(
        self,
        batch: typing.Dict[str, typing.Dict[str, tf.Tensor]],
        carryover: typing.Any = None,
    ) -> tf.Tensor:
        currents = []
        for properties in batch.values():
            vel = properties[mdsuite_properties.velocities.name]
            charge = properties[mdsuite_properties.charge.name]
            currents.append(tf.reduce_sum(charge * vel, axis=0))
        return tf.add_n(currents)
