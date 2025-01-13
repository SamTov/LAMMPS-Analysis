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


class ThermalFlux(MultiSpeciesTrafo):
    """Transformation to calculate the integrated heat current (positions * energies)."""

    def __init__(self):
        super(ThermalFlux, self).__init__(
            input_properties=[
                mdsuite_properties.stress,
                mdsuite_properties.velocities,
                mdsuite_properties.kinetic_energy,
                mdsuite_properties.potential_energy,
            ],
            output_property=mdsuite_properties.thermal_flux,
            scale_function={"linear": {"scale_factor": 5}},
        )

    def transform_batch(
        self,
        batch: typing.Dict[str, typing.Dict[str, tf.Tensor]],
        carryover: typing.Any = None,
    ) -> tf.Tensor:
        fluxes = []
        for properties in batch.values():
            stress = properties[mdsuite_properties.stress.name]
            vel = properties[mdsuite_properties.velocities.name]
            ke = properties[mdsuite_properties.kinetic_energy.name]
            pe = properties[mdsuite_properties.potential_energy.name]
            phi_x = (
                stress[:, :, 0] * vel[:, :, 0]
                + stress[:, :, 3] * vel[:, :, 1]
                + stress[:, :, 4] * vel[:, :, 2]
            )
            phi_y = (
                stress[:, :, 3] * vel[:, :, 0]
                + stress[:, :, 1] * vel[:, :, 1]
                + stress[:, :, 5] * vel[:, :, 2]
            )
            phi_z = (
                stress[:, :, 4] * vel[:, :, 0]
                + stress[:, :, 5] * vel[:, :, 1]
                + stress[:, :, 2] * vel[:, :, 2]
            )

            phi = np.dstack([phi_x, phi_y, phi_z])

            phi_sum_atoms = phi.sum(axis=0)
            # phi_sum_atoms = (
            #         phi_sum_atoms / self.experiment.units["NkTV2p"]
            # )  # factor for units lammps nktv2p
            # TODO why is there a unit conversion in the transformation?

            energy = ke + pe

            energy_velocity = energy * vel
            energy_velocity_atoms = tf.reduce_sum(energy_velocity, axis=0)
            fluxes.append(energy_velocity_atoms - phi_sum_atoms)

        return tf.add_n(fluxes)
