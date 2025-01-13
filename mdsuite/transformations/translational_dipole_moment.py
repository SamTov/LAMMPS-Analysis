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


class TranslationalDipoleMoment(MultiSpeciesTrafo):
    """
    Transformation to compute the translational dipole moment.

    The translational dipole moment is defined as the charges * positions.
    """

    def __init__(self):
        super(TranslationalDipoleMoment, self).__init__(
            input_properties=[
                mdsuite_properties.unwrapped_positions,
                mdsuite_properties.charge,
            ],
            output_property=mdsuite_properties.translational_dipole_moment,
            scale_function={"linear": {"scale_factor": 2}},
        )

    def transform_batch(
        self,
        batch: typing.Dict[str, typing.Dict[str, tf.Tensor]],
        carryover: typing.Any = None,
    ) -> tf.Tensor:
        dipms = []
        for properties in batch.values():
            pos = properties[mdsuite_properties.unwrapped_positions.name]
            charge = properties[mdsuite_properties.charge.name]
            dipms.append(tf.reduce_sum(charge * pos, axis=0))
        return tf.add_n(dipms)
