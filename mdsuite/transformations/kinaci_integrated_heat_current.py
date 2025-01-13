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

from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.transformations.transformations import MultiSpeciesTrafo


class KinaciIntegratedHeatCurrent(MultiSpeciesTrafo):
    """Transformation to calculate the Kinaci integrated heat current."""

    def __init__(self):
        super(KinaciIntegratedHeatCurrent, self).__init__(
            input_properties=[
                mdsuite_properties.unwrapped_positions,
                mdsuite_properties.velocities,
                mdsuite_properties.forces,
                mdsuite_properties.potential_energy,
                mdsuite_properties.time_step,
                mdsuite_properties.sample_rate,
            ],
            output_property=mdsuite_properties.kinaci_heat_current,
            scale_function={"linear": {"scale_factor": 5}},
        )

    def transform_batch(
        self,
        batch: typing.Dict[str, typing.Dict[str, tf.Tensor]],
        carryover: tf.Tensor = None,
    ) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        currents = []

        if carryover is None:
            integrals = []
        else:
            # obtain batch size from the length of the positions array
            batch_size = np.shape(
                list(batch.values())[0][mdsuite_properties.unwrapped_positions.name]
            )[1]
            integrals = [tf.tile(carryover, (1, batch_size))]

        for properties in batch.values():
            pos = properties[mdsuite_properties.unwrapped_positions.name]
            vel = properties[mdsuite_properties.velocities.name]
            force = properties[mdsuite_properties.forces.name]
            pot_energy = properties[mdsuite_properties.potential_energy.name]
            time_step = properties[mdsuite_properties.time_step.name]
            sample_rate = properties[mdsuite_properties.sample_rate.name]

            integrand = tf.einsum("ijk,ijk->ij", force, vel)
            # add here the value from the previous iteration to all the steps in
            # this batch.
            integrals.append(tf.cumsum(integrand, axis=1) * time_step * sample_rate)

            r_k = tf.einsum("ijk,ij->jk", pos, tf.add_n(integrals))
            r_p = tf.einsum("ijk,ijm->jm", pot_energy, pos)

            currents.append(r_k + r_p)

        tot_current = tf.add_n(currents)
        last_integral = tf.add_n(integrals)[:, -1]
        return tot_current, last_integral

    # for reference: the old implementation. Someone who understands the transformation
    # please provide a test

    # def _transformation(
    #     self, data: tf.Tensor, cumul_integral, batch_size
    # ) -> Tuple[tf.Tensor, tf.Tensor]:
    #     """
    #     Calculate the integrated thermal current of the system.
    #
    #     Returns
    #     -------
    #     Integrated heat current : tf.Tensor
    #             The values for the integrated heat current.
    #     """
    #     integral = tf.tile(cumul_integral, (1, batch_size))
    #     system_current = tf.zeros((batch_size, 3), dtype=tf.float64)
    #
    #     for species in self.experiment.species:
    #         positions_path = str.encode(join_path(species, "Unwrapped_Positions"))
    #         velocity_path = str.encode(join_path(species, "Velocities"))
    #         force_path = str.encode(join_path(species, "Forces"))
    #         pe_path = str.encode(join_path(species, "PE"))
    #
    #         integrand = tf.einsum("ijk,ijk->ij", data[force_path], data[velocity_path])
    #         # add here the value from the previous iteration to all the steps in
    #         # this batch.
    #         integral += (
    #             tf.cumsum(integrand, axis=1)
    #             * self.experiment.time_step
    #             * self.experiment.sample_rate
    #         )
    #
    #         r_k = tf.einsum("ijk,ij->jk", data[positions_path], integral)
    #         r_p = tf.einsum("ijk,ijm->jm", data[pe_path], data[positions_path])
    #
    #         system_current += r_k + r_p
    #
    #     cumul_integral = tf.expand_dims(integral[:, -1], axis=1)
    #     return system_current, cumul_integral
