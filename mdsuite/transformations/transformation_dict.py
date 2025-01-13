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

from mdsuite.database.mdsuite_properties import mdsuite_properties as mdp
from mdsuite.transformations import (
    CoordinateUnwrapper,
    CoordinateWrapper,
    IntegratedHeatCurrent,
    IonicCurrent,
    KinaciIntegratedHeatCurrent,
    MomentumFlux,
    ScaleCoordinates,
    ThermalFlux,
    TranslationalDipoleMoment,
    UnwrapViaIndices,
    VelocityFromPositions,
)

# Use this dictionary to determine which property can be obtained by which transformation.
# Needed for transformation dependency resolution.


property_to_transformation_dict = {
    mdp.integrated_heat_current: IntegratedHeatCurrent,
    mdp.ionic_current: IonicCurrent,
    mdp.kinaci_heat_current: KinaciIntegratedHeatCurrent,
    mdp.momentum_flux: MomentumFlux,
    mdp.positions: [
        ScaleCoordinates,
        CoordinateWrapper,
    ],
    mdp.thermal_flux: ThermalFlux,
    mdp.translational_dipole_moment: TranslationalDipoleMoment,
    mdp.unwrapped_positions: [
        UnwrapViaIndices,
        CoordinateUnwrapper,
    ],
    mdp.velocities_from_positions: VelocityFromPositions,
}
