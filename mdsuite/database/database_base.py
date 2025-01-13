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

import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm.session import Session

from .scheme import Base

log = logging.getLogger(__name__)


class DatabaseBase:
    """Docstring."""

    def __init__(self, database_name: str):
        """

        Parameters
        ----------
        database_name: str
            name of the database

        """
        self.name = ""  # Name of the Project
        self.database_name = database_name
        self.storage_path = "./"

        self._engine = None
        self._Session = None

    @property
    def engine(self) -> Engine:
        """Create a SQLAlchemy Engine.

        Returns
        -------
        a SQLAlchemy Engine connected to the Database

        """
        if self._engine is None:
            engine_path = Path(self.storage_path, self.name, self.database_name)
            self._engine = sa.create_engine(
                f"sqlite+pysqlite:///{engine_path}", echo=False, future=True
            )
        return self._engine

    @property
    def session(self) -> Session:
        """

        Notes
        -----
        Use with context manager like:
        >>> with self.session as ses:
        >>>     ses.add()

        Returns
        -------
            Session that can be used inside a context manager

        """
        return sessionmaker(bind=self.engine, future=True)()

    @property
    def base(self) -> declarative_base:
        """

        Returns
        -------
        Get the declarative base from the Database scheme

        """
        return Base

    def build_database(self):
        """Build the database and get create the tables."""
        log.debug("Creating the database if it does not exist.")
        self.base.metadata.create_all(self.engine)
