"""Test MDSuite units."""

import mdsuite as mds


def test_units_loading():
    """Test that the import works."""
    assert mds.units.SI.pressure == 1
    assert mds.units.METAL.pressure == 100000
    assert mds.units.REAL.pressure == 101325.0
