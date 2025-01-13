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
import operator

from mdsuite.calculators.calculator import Calculator, call
from mdsuite.utils.units import boltzmann_constant, elementary_charge

log = logging.getLogger(__name__)


class NernstEinsteinIonicConductivity(Calculator):
    """
    Class for the calculation of the Nernst-Einstein ionic conductivity.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.NernstEinsteinIonicConductivity()

    """

    def __init__(self, **kwargs):
        """
        Standard constructor.

        Parameters
        ----------
        experiment : Experiment
                Experiment class from which to read

        """
        super().__init__(**kwargs)
        self.post_generation = True

        # Properties
        self._truth_table = None

        self.database_group = "Ionic_Conductivity"
        self.analysis_name = "Nernst_Einstein_Ionic_Conductivity"

    @call
    def __call__(
        self,
        corrected: bool = False,
        plot: bool = False,
        data_range: int = 1,
        export: bool = False,
        species: list = None,
        save: bool = True,
    ):
        """
        Standard constructor.

        Parameters
        ----------
        corrected : bool
                If true, correct the output with the distinct diffusion
                coefficient.
        export : bool
                If true, generate a csv file after the analysis.
        plot : bool
                if true, plot the output.
        species : list
                List of species on which to operate.
        data_range : int
                Data range to use in the analysis.
        save : bool
                if true, save the output.

        """
        self.update_user_args(plot=plot, save=False, data_range=data_range, export=export)
        self.corrected = corrected
        self.data = self._load_data()  # tensor_values to be read in

        if species is None:
            self.species = list(self.experiment.species)
        else:
            self.species = species

    def _load_data(self):
        """
        Load tensor_values from a yaml file.

        Returns
        -------
        tensor_values: dict
                A dictionary of tensor_values stored in the yaml file

        """
        test = self.experiment.export_property_data(
            {"property": "Diffusion_Coefficients"}
        )
        return test

    @property
    def truth_table(self):
        """
        Builds a truth table to communicate which tensor_values is available to the
        analysis.

        Returns
        -------
        truth_table : list
                A truth table communication which tensor_values is available for the
                analysis.

        """
        if self._truth_table is None:
            log.warning(
                "No support for different data ranges! This method always picks the"
                " first entry in the database!"
            )
            case_1 = self.experiment.export_property_data(
                {
                    "property": "Diffusion_Coefficients",
                    "analysis": "Green_Kubo_Self_Diffusion_Coefficients",
                }
            )
            case_2 = self.experiment.export_property_data(
                {
                    "property": "Diffusion_Coefficients",
                    "analysis": "Green_Kubo_Distinct_Diffusion_Coefficients",
                }
            )
            case_3 = self.experiment.export_property_data(
                {
                    "property": "Diffusion_Coefficients",
                    "analysis": "Einstein_Self_Diffusion_Coefficients",
                }
            )
            case_4 = self.experiment.export_property_data(
                {
                    "property": "Diffusion_Coefficients",
                    "analysis": "Einstein_Distinct_Diffusion_Coefficients",
                }
            )
            truth_table = [
                list(map(operator.not_, [not case_1, not case_2])),
                list(map(operator.not_, [not case_3, not case_4])),
            ]
            self._truth_table = truth_table
        return self._truth_table

    def _nernst_einstein(self, diffusion_information: list):
        """
        Calculate the Nernst-Einstein ionic conductivity.

        Parameters
        ----------
        diffusion_information : list
                A list of dictionaries loaded from the SQL properties database.

        Returns
        -------
        Nernst-Einstein Ionic conductivity of the experiment in units of S/cm

        """
        # evaluate the prefactor
        numerator = self.experiment.number_of_atoms * (elementary_charge**2)
        denominator = (
            boltzmann_constant
            * self.experiment.temperature
            * (self.experiment.volume * self.experiment.units.volume)
        )
        prefactor = numerator / denominator

        conductivity = 0.0
        uncertainty = 0.0
        for item in diffusion_information:
            log.debug(f"Analysing: {item}")
            diffusion_coefficient = item.data_dict[0].x
            diffusion_uncertainty = item.data_dict[0].uncertainty
            species = item.subjects[0].subject
            charge_term = self.experiment.species[species].charge[0] ** 2
            mass_fraction_term = (
                self.experiment.species[species].n_particles
                / self.experiment.number_of_atoms
            )
            conductivity += diffusion_coefficient * charge_term * mass_fraction_term
            uncertainty += diffusion_uncertainty * charge_term * mass_fraction_term
            data_range = item.data_range

        return [prefactor * conductivity, prefactor * uncertainty, data_range]

    def _corrected_nernst_einstein(
        self, self_diffusion_information: list, distinct_diffusion_information: list
    ):
        """
        Calculate the corrected Nernst-Einstein ionic conductivity.

        Parameters
        ----------
        self_diffusion_information : dict
                dictionary containing information about self diffusion
        distinct_diffusion_information : dict
                dictionary containing information about distinct diffusion

        Returns
        -------
        Corrected Nernst-Einstein ionic conductivity in units of S/cm

        """
        # evaluate the prefactor
        numerator = self.experiment.number_of_atoms * (elementary_charge**2)
        denominator = (
            boltzmann_constant
            * self.experiment.temperature
            * (self.experiment.volume * self.experiment.units.volume)
        )
        prefactor = numerator / denominator

        conductivity = 0.0
        uncertainty = 0.0
        for item in self_diffusion_information:
            diffusion_coefficient = item["data"]
            diffusion_uncertainty = item["uncertainty"]
            species = item["Subject"]
            charge_term = self.experiment.species[species].charge[0] ** 2
            mass_fraction_term = (
                self.experiment.species[species].n_particles
                / self.experiment.number_of_atoms
            )
            conductivity += diffusion_coefficient * charge_term * mass_fraction_term
            uncertainty += diffusion_uncertainty * charge_term * mass_fraction_term
            data_range = item["data_range"]
        for item in distinct_diffusion_information:
            diffusion_coefficient = item["data"]
            diffusion_uncertainty = item["uncertainty"]
            constituents = item["Subject"].split("_")
            charge_term = (
                self.experiment.species[constituents[0]].charge[0]
                * self.experiment.species[constituents[1]].charge[0]
            )
            mass_fraction_term = (
                self.experiment.species[constituents[0]].n_particles
                / self.experiment.number_of_atoms
            ) * (
                self.experiment.species[constituents[1]].n_particles
                / self.experiment.number_of_atoms
            )
            conductivity += diffusion_coefficient * charge_term * mass_fraction_term
            uncertainty += diffusion_uncertainty * charge_term * mass_fraction_term

        return [prefactor * conductivity, prefactor * uncertainty, data_range]

    def _run_nernst_einstein(self):
        """
        Process truth table and run all possible nernst-einstein calculations.

        Returns
        -------

        """
        ne_table = [self.truth_table[0][0], self.truth_table[1][0]]
        if ne_table[0]:
            input_data = [
                self.experiment.export_property_data(
                    {
                        "property": "Diffusion_Coefficients",
                        "analysis": "Green_Kubo_Self_Diffusion_Coefficients",
                        "subjects": [species[0] for species in self.species],
                    }
                )
            ]
            data = self._nernst_einstein(input_data)
            properties = {
                "Property": self.database_group,
                "Analysis": "Green_Kubo_Nernst_Einstein_Ionic_Conductivity",
                "Subject": ["System"],
                "data_range": data[2],
                "data": [{"x": data[0], "uncertainty": data[1]}],
            }
            self._update_properties_file(properties)

        if ne_table[1]:
            input_data = [
                self.experiment.export_property_data(
                    {
                        "property": "Diffusion_Coefficients",
                        "analysis": "Einstein_Self_Diffusion_Coefficients",
                        "subjects": [species[0] for species in self.species],
                    }
                )
            ]
            data = self._nernst_einstein(input_data)

            properties = {
                "Property": self.database_group,
                "Analysis": "Einstein_Nernst_Einstein_Ionic_Conductivity",
                "Subject": ["System"],
                "data_range": data[2],
                "data": [{"x": data[0], "uncertainty": data[1]}],
            }
            self._update_properties_file(properties)

        if not any(ne_table):
            ValueError(
                "There is no values to analyse, please run a diffusion calculation to"
                " proceed"
            )

    def _run_corrected_nernst_einstein(self):
        """
        Process truth table and run all possible nernst-einstein calculations.

        Returns
        -------
        Updates the experiment database_path

        """
        cne_table = [self.truth_table[0][1], self.truth_table[1][1]]

        if cne_table[0]:
            input_self = [
                self.experiment.export_property_data(
                    {
                        "property": "Diffusion_Coefficients",
                        "analysis": "Green_Kubo_Self_Diffusion_Coefficients",
                        "subjects": [species],
                    }
                )[0]
                for species in self.species
            ]
            input_distinct = [
                self.experiment.export_property_data(
                    {
                        "property": "Diffusion_Coefficients",
                        "analysis": "Green_Kubo_Distinct_Diffusion_Coefficients",
                        "subjects": [species],
                    }
                )[0]
                for species in self.species
            ]
            data = self._corrected_nernst_einstein(input_self, input_distinct)

            properties = {
                "Property": self.database_group,
                "Analysis": "Green_Kubo_Corrected_Nernst_Einstein_Ionic_Conductivity",
                "Subject": ["System"],
                "data_range": data[2],
                "data": [{"x": data[0], "uncertainty": data[1]}],
            }
            self._update_properties_file(properties)

        if cne_table[1]:
            input_self = [
                self.experiment.export_property_data(
                    {
                        "property": "Diffusion_Coefficients",
                        "analysis": "Einstein_Self_Diffusion_Coefficients",
                        "subjects": [species],
                    }
                )[0]
                for species in self.species
            ]
            input_distinct = [
                self.experiment.export_property_data(
                    {
                        "property": "Diffusion_Coefficients",
                        "analysis": "Einstein_Distinct_Diffusion_Coefficients",
                        "subjects": [species],
                    }
                )[0]
                for species in self.species
            ]
            data = self._corrected_nernst_einstein(input_self, input_distinct)

            properties = {
                "Property": self.database_group,
                "Analysis": "Einstein_Corrected_Nernst_Einstein_Ionic_Conductivity",
                "Subject": ["System"],
                "data_range": data[2],
                "data": [{"x": data[0], "uncertainty": data[1]}],
            }
            self._update_properties_file(properties)

    def run_post_generation_analysis(self):
        """Run the analysis."""
        self._run_nernst_einstein()
        if self.corrected:
            self._run_corrected_nernst_einstein()
