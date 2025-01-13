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

from dataclasses import dataclass

from mdsuite.database.simulation_database import PropertyInfo


# disable init. This class is not supposed to be changed from its default values
@dataclass(frozen=True, init=False)
class MDSuiteProperties:
    """The names of properties used by MDSuite.

    Use members in the code whenever referencing properties. A string is only needed when
    writing/reading the database.
    Non-obvious members are described below:

    Attributes
    ----------
    scaled_positions:
        Particle positions relative to the box size. All entries are in [0, box_length]
    box_length:
        The lengths of the three sides of the simulation box. Assumes a cuboid.
    time_step: float
        The time step of the simulation. Not to be confused with sample_rate.
    sample_rate: int
        The number of timesteps between successive samples.

    """

    temperature = PropertyInfo("Temperature", 1)
    time = PropertyInfo("Time", 1)
    thermal_flux = PropertyInfo("Thermal_Flux", 3)
    stress_viscosity = PropertyInfo("Stress_Visc", 3)
    momentum_flux = PropertyInfo("Momentum_Flux", 3)
    ionic_current = PropertyInfo("Ionic_Current", 3)
    translational_dipole_moment = PropertyInfo("Translational_Dipole_Moment", 3)
    positions = PropertyInfo("Positions", 3)
    scaled_positions = PropertyInfo("Scaled_Positions", 3)
    box_length = PropertyInfo("Box_Array", 3)  # TODO experiment-wide properties should
    # get their names from here, too
    unwrapped_positions = PropertyInfo("Unwrapped_Positions", 3)
    scaled_unwrapped_positions = PropertyInfo("Scaled_Unwrapped_Positions", 3)
    velocities = PropertyInfo("Velocities", 3)
    velocities_from_positions = PropertyInfo("Velocities_From_Positions", 3)
    momenta = PropertyInfo("Momenta", 3)
    forces = PropertyInfo("Forces", 3)
    box_images = PropertyInfo("Box_Images", 3)
    dipole_orientation_magnitude = PropertyInfo("Dipole_Orientation_Magnitude", 1)
    angular_velocity_spherical = PropertyInfo("Angular_Velocity_Spherical", 3)
    angular_velocity_non_spherical = PropertyInfo("Angular_Velocity_Non_Spherical", 3)
    torque = PropertyInfo("Torque", 3)
    integrated_heat_current = PropertyInfo("Integrated_Heat_Current", 3)
    kinaci_heat_current = PropertyInfo("Kinaci_Heat_Current", 3)
    charge = PropertyInfo("Charge", 1)
    energy = PropertyInfo("Energy", 1)
    kinetic_energy = PropertyInfo("Kinetic_Energy", 1)
    potential_energy = PropertyInfo("Potential_Energy", 1)
    stress = PropertyInfo("Stress", 6)
    time_step = PropertyInfo("Time_Step", 1)
    sample_rate = PropertyInfo("Sample_Rate", 1)


mdsuite_properties = MDSuiteProperties()
