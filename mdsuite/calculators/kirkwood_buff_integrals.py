"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Class for the calculation of the coordinated numbers
"""
import logging
import numpy as np
from mdsuite.database.calculator_database import Parameters
from mdsuite.utils.exceptions import NotApplicableToAnalysis
from mdsuite.calculators.calculator import Calculator, call

log = logging.getLogger(__name__)


class KirkwoodBuffIntegral(Calculator):
    """
    Class for the calculation of the Kikrwood-Buff integrals

    Attributes
    ----------
    experiment : class object
                        Class object of the experiment.
    data_range : int (default=500)
                        Range over which the property should be evaluated. This is not applicable to the current
                        analysis as the full rdf will be calculated.
    x_label : str
                        How to label the x axis of the saved plot.
    y_label : str
                        How to label the y axis of the saved plot.
    analysis_name : str
                        Name of the analysis. used in saving of the tensor_values and figure.
    file_to_study : str
                        The tensor_values file corresponding to the rdf being studied.
    data_files : list
                        list of files to be analyzed.
    rdf = None : list
                        rdf tensor_values being studied.
    radii = None : list
                        radii tensor_values corresponding to the rdf.
    species_tuple : list
                        A list of species combinations being studied.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.KirkwoodBuffIntegral()
    """

    def __init__(self, **kwargs):
        """
        Python constructor for the class

        Parameters
        ----------
        experiment : class object
                        Class object of the experiment.
        """

        super().__init__(**kwargs)
        self.file_to_study = None
        self.data_files = []
        self.rdf = None
        self.radii = None
        self.species_tuple = None
        self.kb_integral = None
        self.database_group = "Kirkwood_Buff_Integral"
        self.x_label = r"r ($\AA$)"
        self.y_label = r"$G(\mathbf{r})$"
        self.analysis_name = "Kirkwood-Buff_Integral"

        self.post_generation = True

    @call
    def __call__(self, plot=True, save=True, data_range=1, export: bool = False):
        """
        Doc string for this one.
        Parameters
        ----------
        plot : bool
                If true, the output will be displayed in a figure. This figure will also be saved.
        save : bool
                If true, the data from the analysis will be saved in the sql database
        data_range : int
                Default to 1 for this analysis
        export : bool
                If tue, csv files will be dumped after the analysis.
        """

        self.update_user_args(
            plot=plot, save=save, data_range=data_range, export=export
        )

    def _autocorrelation_time(self):
        """
        Not needed in this analysis
        """
        raise NotApplicableToAnalysis

    def _calculate_kb_integral(self):
        """
        calculate the Kirkwood-Buff integral
        """

        self.kb_integral = []  # empty the integration tensor_values

        for i in range(1, len(self.radii)):
            self.kb_integral.append(
                4
                * np.pi
                * (
                    np.trapz(
                        (self.rdf[1:i] - 1) * (self.radii[1:i]) ** 2, x=self.radii[1:i]
                    )
                )
            )

    def run_post_generation_analysis(self):
        """
        Calculate the potential of mean-force and perform error analysis
        """
        calculations = self._get_rdf_data()
        for i, data in enumerate(calculations):  # Loop over all existing RDFs
            if i == len(calculations) - 1:
                self.last_iteration = True
            self.species_tuple = "_".join(data.subjects)
            self.data_range = data.data_range
            self._load_rdf_from_file(data)  # load the tensor_values from it
            self._calculate_kb_integral()  # Integrate the rdf and calculate the KB integral

            data = [
                {"r": x, "kb_integral": y} for x, y in
                zip(self.radii[1:], self.kb_integral)
            ]
            log.debug(f"Writing {self.analysis_name} to database!")
            properties = Parameters(
                Property=self.database_group,
                Analysis=self.analysis_name,
                data_range=self.data_range,
                data=data,
                Subject=[self.species_tuple]
            )
            self.update_database(properties)

            # Plot if required
            if self.plot:
                self.run_visualization(
                    x_data=self.radii[1:],
                    y_data=self.kb_integral,
                    title=f"{self.species_tuple}",
                )
