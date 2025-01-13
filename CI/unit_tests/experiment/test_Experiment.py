"""Test MDSuite Experiment class."""

import pytest

from mdsuite.experiment.experiment import Experiment


def test_experiment_name():
    with pytest.raises(ValueError):
        Experiment(project=None, name="250K")
