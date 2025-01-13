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

import numpy as np
import tensorflow as tf

from mdsuite.transformations.transformations import (
    MultiSpeciesTrafo,
    SingleSpeciesTrafo,
)


class TestMultispecies(MultiSpeciesTrafo):
    """transformation for testing purposes only."""

    def __init__(self, input_properties: typing.Iterable, output_property):
        super(TestMultispecies, self).__init__(
            input_properties=input_properties,
            output_property=output_property,
            scale_function={"linear": {"scale_factor": 2}},
        )

    def transform_batch(
        self,
        batch: typing.Dict[str, typing.Dict[str, tf.Tensor]],
        carryover: typing.Any = None,
    ) -> typing.Tuple[tf.Tensor, typing.Any]:
        """'use' all input properties, return array of 516."""
        for properties in batch.values():
            for input_prop in self.input_properties:
                val = properties[input_prop.name]
                assert val is not None

        assert carryover is None or carryover == 17

        sp_name = list(batch.keys())[0]
        shape = np.shape(batch[sp_name][self.input_properties[0].name])
        out = tf.fill(
            (1, shape[1], self.output_property.n_dims), tf.cast(516, self.dtype)
        )
        return out, 17


class TestSingleSpecies(SingleSpeciesTrafo):
    """Wrap coordinates into the simulation box."""

    def __init__(self, input_properties: typing.Iterable, output_property):
        super(TestSingleSpecies, self).__init__(
            input_properties=input_properties,
            output_property=output_property,
            scale_function={"linear": {"scale_factor": 2}},
        )

    def transform_batch(
        self, batch: typing.Dict[str, tf.Tensor], carryover: typing.Any = None
    ) -> typing.Tuple[tf.Tensor, int]:
        """'use' all input properties, return array of 516."""
        self.logger.debug(batch)
        for input_prop in self.input_properties:
            val = batch[input_prop.name]
            assert val is not None

        assert carryover is None or carryover == 17

        shape = np.shape(batch[self.input_properties[0].name])
        out = tf.fill(
            (1, shape[1], self.output_property.n_dims), tf.cast(516, self.dtype)
        )
        return out, 17
