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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdsuite import Experiment


class RunModule:
    """Run a calculator from the experiment class.

    Notes
    -----
    This class is a helper to convert the dictionary of possible computations
    "dict_classes_computations" into attributes of the `experiment.run_computation`
    helper class.

    """

    def __init__(self, parent, module_dict, **kwargs):
        """Initialize the attributes

        Parameters
        ----------
        parent: Experiment
            the experiment to be passed to the calculator afterwards
        module_dict: dict
            A dictionary containing all the modules / calculators / Time series
            operations with their names as keys
        kwargs:
            Additional parameters to be passed to the module_dict.

        """
        self.parent: Experiment = parent
        self._kwargs = kwargs
        self._module_dict = module_dict
        for key in self._module_dict:
            self.__setattr__(key, self._module_dict[key])

    def __getattribute__(self, item):
        """Call via function
        You can call the computation via a function and autocompletion
        >>> self.run.EinsteinDiffusionCoefficients(plot=True).

        Returns
        -------
            Instantiated calculator class with added experiment that can be called.

        """
        if item.startswith("_"):
            # handle private functions
            return super().__getattribute__(item)

        try:
            class_compute = self._module_dict[item]
        except KeyError:
            return super().__getattribute__(item)

        return class_compute(experiment=self.parent, **self._kwargs)
