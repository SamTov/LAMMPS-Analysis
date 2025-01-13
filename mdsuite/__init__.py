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

try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata

import logging
import sys

from mdsuite import utils
from mdsuite.experiment import Experiment
from mdsuite.project import Project
from mdsuite.utils import config, units
from mdsuite.utils.molecule import Molecule
from mdsuite.utils.report_computer_characteristics import Report

__all__ = [
    Project.__name__,
    Experiment.__name__,
    Report.__name__,
    "config",
    Molecule.__name__,
    "units",
    "utils",
]
__version__ = metadata.version("mdsuite")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Formatter for advanced logging
formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

channel = logging.StreamHandler(sys.stdout)
channel.setLevel(logging.INFO)
channel.setFormatter(formatter)

logger.addHandler(channel)
