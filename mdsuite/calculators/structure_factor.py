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
Code for the computation of the structure factor.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pkg_resources
import tensorflow as tf
from bokeh.models import HoverTool
from bokeh.plotting import figure

from mdsuite import data, utils
from mdsuite.calculators.calculator import Calculator, call
from mdsuite.database.scheme import Computation

log = logging.getLogger(__name__)


@dataclass
class Args:
    """Data class for the saved properties."""

    number_of_bins: int
    number_of_configurations: int
    cutoff: float
    resolution: int


@dataclass
class SpeciesData:
    """Data class for species data to be used in the calculation."""

    particle_density: float
    molar_fraction: float
    form_factor: np.ndarray = None


class StructureFactor(Calculator):
    """
    Class for the calculation of the total structure factor for X-ray scattering
    using the Faber-Ziman partial structure factors. This analysis is valid for a
    magnitude of the X-ray scattering vector Q < 25 * 1/Angstrom. This means that
    the radii of the rdf has to be in Angstrom, otherwise it wont work.
    Explicitly equations 9, 10 and 11 of the paper.

    'DFT Accurate Interatomic Potential for Molten NaCl from MachineLearning' from
    Samuel Tovey, Anand Narayanan Krishnamoorthy, Ganesh Sivaraman, Jicheng Guo,
    Chris Benmore,Andreas Heuer, and Christian Holm are implemented.

    Attributes
    ----------
    experiment : class object
                            Class object of the experiment.
    x_label : str
                            How to label the x axis of the saved plot.
    y_label : str
                        How to label the y axis of the saved plot.
    analysis_name : str
                        Name of the analysis. used in saving of the tensor_values and
                        figure.
    file_to_study : str
                        The tensor_values file corresponding to the rdf being studied.
    data_files : list
                        list of files to be analyzed.
    rdf = None : list
                        rdf tensor_values being studied.
    radii = None : list
                        radii tensor_values corresponding to the rdf.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.StructureFactor()

    Notes
    -----
    In order to use the structure factor calculator both the masses and the
    charges of each species must be present. If they are not correct, the structure
    factor will not work.

    """

    volume: float
    species_dict: dict
    number_of_atoms: int
    total_density: float

    def __init__(self, **kwargs):
        """
        Constructor for the class.

        Parameters
        ----------
        experiment : class object
                Class object of the experiment.

        """
        super().__init__(**kwargs)

        self.post_generation = True

        self.x_label = r"$$\text{Q} / nm ^{-1}$$"
        self.y_label = r"$$\text{S(Q)}$$"
        self.analysis_name = "total_structure_factor"

        self.result_series_keys = ["q", "S"]

        # Read the data from the file.
        stream = pkg_resources.resource_stream(data.__name__, "form_fac_coeffs.csv")
        self.form_factor_data = pd.read_csv(stream)

    @call
    def __call__(
        self,
        rdf_data: Computation = None,
        plot=True,
        method: str = "Faber-Ziman",
        resolution: int = 700,
    ):
        """
        Parameters
        ----------
        rdf_data : Computation (optional)
                MDSuite Computation data schema from which to load the RDF data and
                store relevant SQL meta-data information. If not give, an RDF will be
                computed using the default RDF arguments.
        plot : bool (default=True)
                Decision to plot the analysis.
        method : str (default=Faber-Ziman)
                Method use to compute the weight factors.
        resolution : int (default=700)
                Resolution of the structure factor.

        """
        self.plot = plot

        if isinstance(rdf_data, Computation):
            self.rdf_data = rdf_data
        else:
            self.rdf_data = self.experiment.run.RadialDistributionFunction(plot=False)

        # set args that will affect the computation result
        self.args = Args(
            number_of_bins=self.rdf_data.computation_parameter["number_of_bins"],
            cutoff=self.rdf_data.computation_parameter["cutoff"],
            number_of_configurations=self.rdf_data.computation_parameter[
                "number_of_configurations"
            ],
            resolution=resolution,
        )
        self.q_values = np.linspace(0.5, 12, resolution)

        self._compute_angstsrom_volume()  # correct the volume of the system.
        self.number_of_atoms = sum(
            [species.n_particles for species in self.experiment.species.values()]
        )
        self.total_density = self.number_of_atoms / self.volume

        # Construct the species dictionary and fill all species information.
        self.species_dict = {}
        for species in self.experiment.species:
            n_particles = self.experiment.species[species].n_particles
            particle_density = n_particles / self.volume
            molar_fraction = n_particles / self.number_of_atoms
            self.species_dict[species] = SpeciesData(
                particle_density=particle_density,
                molar_fraction=molar_fraction,
            )
        self._compute_form_factors()

    def _compute_angstsrom_volume(self):
        """
        Compute the volume of the box in Angstrom.

        The data for scattering features is in Angstrom. Furthermore, the radial
        component of the calculator is integrated over.

        Returns
        -------
        Updates the volume attribute of the class.

        """
        volume_si = self.experiment.volume * self.experiment.units.volume

        self.volume = volume_si / 1e-10**3

    def _compute_form_factors(self):
        """
        Compute the atomic form factors for each species and add them to the
        data class.

        Returns
        -------
        Updates the data class for each species.

        Notes
        -----
        aff -> atomic form factor

        """
        for name, species_data in self.species_dict.items():
            # aff -> atomic form factor
            aff_data = self.form_factor_data.loc[self.form_factor_data["Element"] == name]
            c = aff_data["c"]
            form_factor = np.zeros(self.args.resolution)
            for i in range(4):
                a = aff_data[f"a{i + 1}"]
                b = aff_data[f"b{i + 1}"]
                form_factor += float(a) * np.exp(
                    -1 * float(b) * (self.q_values / (4 * np.pi))
                ) + float(c)

            species_data.form_factor = form_factor

    def _compute_partial_structure_factors(self) -> dict:
        """
        Compute the partial structure factors.

        Perform a custom fourier transform over the RDF.

        Returns
        -------
        partial_structure_factors : dict
                A dictionary of partial structure factors.

        Notes
        -----
        This expands a tensor by use of an outer produce and therefore could
        theoretically result in memory issue for very large radii values over very
        fine fourier grids. In this case, batching can be performed over Q values.

        """
        partial_structure_factors = {}
        for pair, pair_data in self.rdf_data.data_dict.items():
            radii = np.array(pair_data["x"])[1:] * 10  # convert to Angstrom.
            rdf = np.array(pair_data["y"][1:])
            radial_multiplier = tf.einsum("i, j -> ij", self.q_values, radii)
            pre_factor = radii**2 * np.sin(radial_multiplier) / radial_multiplier
            integral = 1 + 4 * np.pi * np.trapz(y=pre_factor * (rdf - 1), x=radii, axis=1)
            partial_structure_factors[pair] = integral * 0.5

        return partial_structure_factors

    def _compute_weight_factors(self) -> dict:
        """
        Compute the weight factors for the SF computation.

        Compute the weight factors from the atomic form factors (aff) for each species.

        Returns
        -------
        weight_factors : dict
                A dict of weight factors to be used in the SF computation. There is
                one weight factor for each pair.

        """
        weight_factors = {}
        for pair, pair_data in self.rdf_data.data_dict.items():
            species_names = pair.split("_")
            form_factors = [self.species_dict[item].form_factor for item in species_names]
            mean_square_form_factor = np.mean(form_factors) ** 2
            molar_fraction = np.prod(
                [self.species_dict[item].molar_fraction for item in species_names]
            )
            weight_factors[pair] = (
                molar_fraction * np.prod(form_factors) / mean_square_form_factor
            )

        return weight_factors

    def _compute_total_structure_factor(
        self, partial_sf: dict, weight_factors: dict
    ) -> np.ndarray:
        """
        Compute the total structure factor.

        Parameters
        ----------
        partial_sf : dict
                A dict of partial structure factors for each pair.
        weight_factors : dict
                A dict of weight factors computed using the relevant formalism for
                each pair.

        Returns
        -------
        total_structure_factor : np.ndarray
                Total structure factor of the system.

        """
        structure_factor = np.zeros(self.args.resolution)
        for pair, pair_data in partial_sf.items():
            species = pair.split("_")
            if species[0] == species[1]:
                factor = 2
            else:
                factor = 1
            structure_factor += factor * weight_factors[pair] * pair_data

        return structure_factor

    def run_calculator(self):
        """Compute the total structure factor."""
        partial_sf = self._compute_partial_structure_factors()
        print(partial_sf)
        weight_factors = self._compute_weight_factors()
        total_structure_factor = self._compute_total_structure_factor(
            partial_sf, weight_factors
        )

        data_dict = {
            self.result_series_keys[0]: self.q_values.tolist(),
            self.result_series_keys[1]: total_structure_factor.tolist(),
        }

        # Store the total SF data.
        self.queue_data(data=data_dict, subjects=["System"])

        # Store the partial SF data.
        for pair, pair_data in partial_sf.items():
            data_dict = {
                self.result_series_keys[0]: self.q_values.tolist(),
                self.result_series_keys[1]: pair_data.tolist(),
            }
            self.queue_data(data=data_dict, subjects=pair)

    def plot_data(self, data):
        """
        Plot the structure factor data.

        This method will plot both the partial and total structure factor data.

        Parameters
        ----------
        data : data_dict
                Data dict from the calculator over which to loop and the data withing
                plot.

        Returns
        -------

        """
        for key, val in data.items():
            fig = figure(x_axis_label=self.x_label, y_axis_label=self.y_label)
            fig.line(
                val[self.result_series_keys[0]],
                val[self.result_series_keys[1]],
                color=utils.Colour.PRUSSIAN_BLUE,
                legend_label=f"{key}",
            )
            fig.add_tools(HoverTool())

            self.plot_array.append(fig)
