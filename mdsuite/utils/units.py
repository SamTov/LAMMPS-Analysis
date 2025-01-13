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

standard_state_pressure = 100000  # Pa -- Standard state pressure
avogadro_constant = 6.02214076e23  # mol^-1 -- Avogadro's constant
elementary_charge = 1.602176634e-19  # C -- Elementary charge
boltzmann_constant = 1.380649e-23  # J K^-1 --  Boltzmann constant
hyperfine_transition_frequency = (
    9192631770  # Hz -- Hyperfine transition frequency of Cs-133
)
luminous_efficacy = 683  # lm W^-1 -- Luminous efficacy
planck_constant = 6.62607015e-34  # J Hz^-1 -- Planck constant
reduced_planck_constant = 1.054571817e-34  # J s -- Reduced Planck constant
speed_of_light = 299792458  # m s^-1 -- Speed of light in a vacuum
gravity = 9.80665  # m s^-2 -- Standard acceleration due to gravity on earth
atmosphere = 101325  # Pa -- Standard atmospheric pressure
golden_ratio = 1.618033988749895  # The golden ratio as taken from scipy


@dataclass(frozen=True)
class Units:
    """Dataclass for the units."""

    time: float
    length: float
    energy: float
    NkTV2p: float
    boltzmann: float
    temperature: float
    pressure: float
    avogadro: float = 6.02214076e23
    elementary_charge: float = 1.602176634e-19

    @property
    def volume(self) -> float:
        """Get the volume factor."""
        return self.length**3


REAL = Units(
    time=1e-15,
    length=1e-10,
    energy=4184 / 6.02214076e23,
    NkTV2p=68568.415,
    boltzmann=0.0019872067,
    temperature=1,
    pressure=101325.0,
)


METAL = Units(
    time=1e-12,
    length=1e-10,
    energy=1.6022e-19,
    NkTV2p=1.6021765e6,
    boltzmann=8.617343e-5,
    temperature=1,
    pressure=100000,
)


SI = Units(
    time=1,
    length=1,
    energy=1,
    NkTV2p=1.380649e-23,
    boltzmann=1.386049e-23,
    temperature=1,
    pressure=1,
)


units_dict = {"real": REAL, "metal": METAL, "si": SI}
