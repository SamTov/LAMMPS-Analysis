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

import logging
from pathlib import Path

from mdsuite.utils.database import get_or_create

from .database_base import DatabaseBase
from .scheme import Experiment, Project

log = logging.getLogger(__name__)


class ProjectDatabase(DatabaseBase):
    """Class for the management of the project database."""

    def __init__(self):
        """Constructor for the Project database class."""
        super().__init__(database_name="project.db")

    @property
    def project_id(self) -> int:
        """The id of this project in the database."""
        return 1

    @property
    def db_experiments(self):
        """Get all experiments."""
        # renamed to db_experiments because experiments contains the instances of the
        # Experiment class
        with self.session as ses:
            experiments = ses.query(Experiment).all()
        return experiments

    @property
    def description(self):
        with self.session as ses:
            project = get_or_create(ses, Project, id=self.project_id)
            description = project.description
            ses.commit()

        return description

    @description.setter
    def description(self, value: str):
        """
        Allow users to add a short description to their project.

        Parameters
        ----------
        value : str
                Description of the project. If the string ends in .txt, the contents of
                the txt file will be read. If it ends in .md, same outcome. Anything
                else will be read as is.

        """
        if value is None:
            return
        if Path(value).exists():
            value = Path(value).read_text()

        with self.session as ses:
            project = get_or_create(ses, Project, id=self.project_id)
            project.description = value
            ses.commit()
