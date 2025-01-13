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
Unit tests for the molecule data class.
"""

import pytest

import mdsuite
from mdsuite.utils.molecule import Molecule


def test_instantiation():
    """
    Test that the class is instantiated correctly.

    Notes
    -----
    Test the following cases:

    * Fail when essential data is not provided.
    * Store the correct information when provided.
    * Use the correct defaults.

    """
    with pytest.raises(TypeError):
        Molecule()
        Molecule(name="test")
        Molecule(name="test", amount=2)
        Molecule(name="test", cutoff=4)

    my_molecule = Molecule(name="test", amount=2, cutoff=4)

    assert my_molecule.reference_configuration_idx == 0
    assert my_molecule.smiles is None
    assert my_molecule.species_dict is None


def test_project_import():
    """Test that the molecule can be import directly from the mdsuite import."""
    mdsuite.Molecule(name="test", amount=1, cutoff=1.0)
