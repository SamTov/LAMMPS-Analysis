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
from pathlib import Path
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


def test_project_description():
    """Test that the project description is stored correctly in the database."""
    project_1 = mds.Project()
    project_1.description = "HelloWorld"

    project_2 = mds.Project()
    assert project_2.description == "HelloWorld"


def test_project_description_from_file():
    """
    Test that the project description is stored correctly in the database if
    read from file.
    """
    desc = Path("desc.md")
    desc.write_text("HelloWorld")

    project_1 = mds.Project()
    project_1.description = "desc.md"

    project_2 = mds.Project()
    assert project_2.description == "HelloWorld"
