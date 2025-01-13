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

import dataclasses


class _FrozenCls(type):
    def __setattr__(cls, name, value):
        raise dataclasses.FrozenInstanceError(f"cannot assign to attribute '{name}'")


class DatasetKeys(metaclass=_FrozenCls):
    """Class to hold the keys for the datasets."""

    OBSERVABLES = "Observables"
