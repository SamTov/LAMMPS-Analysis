"""MDSuite test for run computation.

This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0.

Copyright Contributors to the Zincware Project.

Description:
"""

from unittest.mock import Mock

from mdsuite.experiment.run import RunComputation


class MockTransformation:
    """Mock a transformation."""

    def run_transformation(self, *args, **kwargs):
        pass


class MockExperiment(Mock):
    """Mock a Experiment class with cls_transformation_run."""

    def cls_transformation_run(self, cls, *args, **kwargs):
        self.cls = cls


def test_transformation_wrapper():
    """Test the transformation wrapper for a single experiment."""
    mock_experiment = MockExperiment()
    run_computation = RunComputation(experiment=mock_experiment)

    wrapped_transformation = run_computation.transformation_wrapper(MockTransformation)
    wrapped_transformation("arg_test", arg2="kwarg_test")

    assert isinstance(mock_experiment.cls, MockTransformation)


def test_transformation_wrapper_multi_exp():
    """Test the transformation wrapper for a single experiment."""
    mock_experiment1 = MockExperiment()
    mock_experiment2 = MockExperiment()
    run_computation = RunComputation(experiments=[mock_experiment1, mock_experiment2])

    wrapped_transformation = run_computation.transformation_wrapper(MockTransformation)
    wrapped_transformation("arg_test", arg2="kwarg_test")

    assert isinstance(mock_experiment1.cls, MockTransformation)
    assert isinstance(mock_experiment2.cls, MockTransformation)
