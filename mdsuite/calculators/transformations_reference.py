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
from mdsuite.transformations.integrated_heat_current import IntegratedHeatCurrent
from mdsuite.transformations.ionic_current import IonicCurrent
from mdsuite.transformations.kinaci_integrated_heat_current import (
    KinaciIntegratedHeatCurrent,
)
from mdsuite.transformations.momentum_flux import MomentumFlux
from mdsuite.transformations.scale_coordinates import ScaleCoordinates
from mdsuite.transformations.thermal_flux import ThermalFlux
from mdsuite.transformations.translational_dipole_moment import (
    TranslationalDipoleMoment,
)
from mdsuite.transformations.unwrap_coordinates import CoordinateUnwrapper
from mdsuite.transformations.unwrap_via_indices import UnwrapViaIndices
from mdsuite.transformations.wrap_coordinates import CoordinateWrapper

switcher_transformations = {
    "Translational_Dipole_Moment": "TranslationalDipoleMoment",
    "Ionic_Current": "IonicCurrent",
    "Integrated_Heat_Current": "IntegratedHeatCurrent",
    "Thermal_Flux": "ThermalFlux",
    "Momentum_Flux": "MomentumFlux",
    "Kinaci_Heat_Current": "KinaciIntegratedHeatCurrent",
}
