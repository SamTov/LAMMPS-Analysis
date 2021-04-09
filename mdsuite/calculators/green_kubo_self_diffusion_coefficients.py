"""
Class for the calculation of the Green-Kubo diffusion coefficients.
Summary
-------
This module contains the code for the Einstein diffusion coefficient class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.einstein_diffusion_coefficients method.
The methods in class can then be called by the Experiment.green_kubo_diffusion_coefficients method and all necessary
calculations performed.
"""

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import warnings
from tqdm import tqdm
import tensorflow as tf
from mdsuite.calculators.calculator import Calculator

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboSelfDiffusionCoefficients(Calculator):
    """
    Class for the Green-Kubo diffusion coefficient implementation
    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    plot : bool
            if true, plot the tensor_values
    species : list
            Which species to perform the analysis on
    data_range :
            Number of configurations to use in each ensemble
    save :
            If true, tensor_values will be saved after the analysis
    x_label : str
            X label of the tensor_values when plotted
    y_label : str
            Y label of the tensor_values when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database_path for the analysis
    correlation_time : int
            Correlation time of the property being studied. This is used to ensure ensemble sampling is only performed
            on uncorrelated samples. If this is true, the error extracted form the calculation will be correct.
    """

    def __init__(self, experiment, plot: bool = False, species: list = None, data_range: int = 500, save: bool = True,
                 correlation_time: int = 1, atom_selection=np.s_[:]):
        """
        Constructor for the Green Kubo diffusion coefficients class.

        Attributes
        ----------
        experiment :  object
                Experiment class to call from
        plot : bool
                if true, plot the tensor_values
        species : list
                Which species to perform the analysis on
        data_range :
                Number of configurations to use in each ensemble
        save :
                If true, tensor_values will be saved after the analysis
        """

        super().__init__(experiment, plot, save, data_range, correlation_time=correlation_time,
                         atom_selection=atom_selection)

        self.loaded_property = 'Velocities'  # Property to be loaded for the analysis
        self.scale_function = {'linear': {'scale_factor': 5}}

        self.species = species  # Which species to calculate for

        self.database_group = 'diffusion_coefficients'  # Which database_path group to save the tensor_values in
        self.x_label = 'Time $(s)$'
        self.y_label = 'VACF $(m^{2}/s^{2})$'
        self.analysis_name = 'Green_Kubo_Diffusion'

        self.vacf = np.zeros(self.data_range)
        self.sigma = []

        if self.species is None:
            self.species = list(self.experiment.species)

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------
        Update the class state.
        """
        self.batch_output_signature = tf.TensorSpec(shape=(None, self.batch_size, 3), dtype=tf.float64)
        self.ensemble_output_signature = tf.TensorSpec(shape=(None, self.data_range, 3), dtype=tf.float64)

    def _calculate_prefactor(self, species: str = None):
        """
        Compute the prefactor

        Parameters
        ----------
        species : str
                Species being studied.

        Returns
        -------
        Updates the class state.
        """
        # Calculate the prefactor
        numerator = self.experiment.units['length'] ** 2
        denominator = 3 * self.experiment.units['time'] * (self.data_range - 1) *\
                      len(self.experiment.species[species]['indices'])
        self.prefactor = numerator / denominator

    def _apply_averaging_factor(self):
        """
        Apply the averaging factor to the msd array.
        Returns
        -------

        """
        self.vacf /= max(self.vacf)

    def _apply_operation(self, ensemble, index):
        """
        Calculate and return the msd.

        Parameters
        ----------
        ensemble

        Returns
        -------
        MSD of the tensor_values.
        """
        vacf = np.zeros(2*self.data_range - 1)
        for item in ensemble:
            vacf += sum([signal.correlate(item[:, idx], item[:, idx], mode="full", method='auto') for idx in range(3)])

        self.vacf += vacf[int(self.data_range - 1):]  # Update the averaged function
        self.sigma.append(np.trapz(vacf[int(self.data_range - 1):], x=self.time))

    def _post_operation_processes(self, species: str = None):
        """
        Apply post-op processes such as saving and plotting.
        Returns
        -------

        """

        result = self.prefactor * np.array(self.sigma)

        self._update_properties_file(item='Singular', sub_item=species,
                                     data=[np.mean(result), np.std(result) / (np.sqrt(len(result)))])
        # Update the plot if required
        if self.plot:
            plt.plot(np.array(self.time) * self.experiment.units['time'], self.vacf, label=species)

        # Save the array if required
        if self.save:
            self._save_data(f"{species}_{self.analysis_name}", [self.time, self.vacf])
