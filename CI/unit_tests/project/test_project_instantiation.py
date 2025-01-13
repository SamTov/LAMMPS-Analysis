"""MDSuite test for project instatiating.

This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0.

Copyright Contributors to the Zincware Project.

Description:
"""

import pathlib

import mdsuite


def test_storage_path_as_string(tmp_path):
    _ = mdsuite.Project(name="project", storage_path=tmp_path.as_posix())
    assert (tmp_path / "project").exists()


def test_storage_path_as_pathlib(tmp_path):
    _ = mdsuite.Project(name="project", storage_path=tmp_path)
    assert isinstance(tmp_path, pathlib.Path)
    assert (tmp_path / "project").exists()
