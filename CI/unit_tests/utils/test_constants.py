"""Test for MDSuite utils.constants."""

import dataclasses

import pytest

import mdsuite as mds


def test_DatasetKeys():
    assert mds.utils.DatasetKeys.OBSERVABLES == "Observables"
    with pytest.raises(dataclasses.FrozenInstanceError):
        mds.utils.DatasetKeys.OBSERVABLES = None
