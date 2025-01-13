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
Module for the MDSuite molecule dataclass
"""

from dataclasses import dataclass


@dataclass
class Molecule:
    """
    Data class to define a molecule.

    Attributes
    ----------
    name : str
            Name of the molecule. This name will be stored in the database.
    smiles : str (optional)
            SMILES string to use in the definition of the molecule internally.
            e.g. CCN1C=C[N+](+C1)C
    species_dict : dict (optional)
            A species dict for a custom molecule in the case where a SMILES string
            cannot be written.
            e.g. {'C': 6, 'N': 2, 'H': 12}
    amount : int
            Number of molecules of this species in the trajectory.
    reference_configuration_idx : int (default=0)
            A specific configuration to use in the construction of the molecules.
    cutoff : float
            A cutoff value to use when identifying bonded pairs. Should be the largest
            bond distance in the system, perhaps with some buffer depending on the
            flexibility of bonds in the molecule and their distribution in the reference
            configuration.
    mol_pbc : bool
            If true, the simulation that was run was using molecule-based PBC, i.e.
            molecules were not allowed to break in the simulation.

    """

    name: str
    amount: int
    cutoff: float
    smiles: str = None
    species_dict: dict = None
    reference_configuration_idx: int = 0
    mol_pbc: bool = False
