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

import os
from tempfile import TemporaryDirectory

import pytest

import mdsuite as mds

temp_dir = TemporaryDirectory()
cwd = os.getcwd()


@pytest.fixture(autouse=True)
def prepare_env():
    """Prepare temporary environment."""
    temp_dir = TemporaryDirectory()
    os.chdir(temp_dir.name)

    yield

    os.chdir(cwd)
    temp_dir.cleanup()


def test_project_load_experiments():
    """Test that the project loads experiments."""
    project_1 = mds.Project()
    project_1.description = "HelloWorld"
    project_1.add_experiment("Exp01", active=False)
    project_1.add_experiment("Exp02", active=False)

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")

    assert len(project_2.active_experiments) == 1

    project_3 = mds.Project()
    project_3.load_experiments(["Exp01", "Exp02"])

    assert len(project_3.active_experiments) == 2
