"""
Class for the calculation of the radial distribution function.

Author: Samuel Tovey, Fabian Zills

Summary
-------
This module contains the code for the radial distribution function. This class is called by
the Experiment class and instantiated when the user calls the Experiment.radial_distribution_function method.
The methods in class can then be called by the Experiment.radial_distribution_function method and all necessary
calculations performed.
"""
from abc import ABC

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import user packages
from tqdm import tqdm
import tensorflow as tf
import itertools

from mdsuite.calculators.calculator import Calculator
from mdsuite.utils.meta_functions import split_array

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class RadialDistributionFunction(Calculator, ABC):
    """
    Class for the calculation of the radial distribution function

    Attributes
    ----------
    obj :  object
            Experiment class to call from
    plot : bool
            if true, plot the data
    data_range :
            Number of configurations to use in each ensemble
    save :
            If true, data will be saved after the analysis
    x_label : str
            X label of the data when plotted
    y_label : str
            Y label of the data when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database for the analysis
    correlation_time : int
            Correlation time of the property being studied. This is used to ensure ensemble sampling is only performed
            on uncorrelated samples. If this is true, the error extracted form the calculation will be correct.
    """

    def __init__(self, obj, plot=True, number_of_bins=None, cutoff=None, save=True, data_range=1, x_label=r'r ($\AA$)',
                 y_label='g(r)', analysis_name='radial_distribution_function', images=1, start=0, stop=None,
                 number_of_configurations=1000):
        """

        Attributes
        ----------
        obj :  object
                Experiment class to call from
        plot : bool
                if true, plot the data
        data_range :
                Number of configurations to use in each ensemble
        save :
                If true, data will be saved after the analysis
        x_label : str
                X label of the data when plotted
        y_label : str
                Y label of the data when plotted
        analysis_name : str
                Name of the analysis
        """
        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name)
        self.parent = obj  # Experiment class to update
        self.number_of_bins = number_of_bins  # Number of number_of_bins to use in the histogram
        self.cutoff = cutoff  # Cutoff for the RDF
        self.correlation_time = None  # Correlation time -- not relevant here
        self.loaded_property = 'Positions'  # Which database property to load
        self.database_group = 'radial_distribution_function'  # Which database group to save the data in

        self.images = images  # number of images to include
        self.start = start  # Which configuration to start at
        self.stop = stop  # Which configuration to stop at
        self.number_of_configurations = number_of_configurations  # Number of configurations to use

        # Perform checks
        if stop is None:
            self.stop = obj.number_of_configurations

        if self.cutoff is None:
            self.cutoff = self.parent.box_array[0] / 2  # set cutoff to half box size if none set

        if self.number_of_bins is None:
            self.number_of_bins = self.cutoff / 0.01  # default is 1/100th of an angstrom

        # Set calculation specific parameters
        self.bin_range = [0, self.cutoff]  # set the bin range
        self.index_list = [i for i in range(len(self.parent.species.keys()))]  # Get the indices of the species
        self.sample_configurations = np.linspace(self.start,
                                                 self.stop,
                                                 self.number_of_configurations,
                                                 dtype=np.int)  # choose sampled configurations
        self.key_list = [self._get_species_names(x) for x in
                         list(itertools.combinations_with_replacement(self.index_list, r=2))]  # Select combinations
        self.rdf = {name: np.zeros(self.number_of_bins) for name in self.key_list}  # instantiate the rdf tuples

    def _autocorrelation_time(self):
        """ Calculate the position autocorrelation time of the system """
        raise NotImplementedError

    def _get_ideal_gas_probability(self):
        """
        Get the correct ideal gas term

        In the case of a cutoff value greater than half of the box size, the ideal gas term of the system must be
        corrected due to the lack of spherical symmetry in the system.
        Returns
        -------
        """

        def _spherical_symmetry(data: np.array):
            """
            Operation to perform for full spherical symmetry

            Parameters
            ----------
            data : np.array
                    data on which to operate
            Returns
            -------
            function_values : np.array
                    result of the operation
            """
            return 4*np.pi*(data**2)

        def _correction_1(data: np.array):
            """
            First correction to ideal gas.

            data : np.array
                    data on which to operate
            Returns
            -------
            function_values : np.array
                    result of the operation

            """

            return 2*np.pi*data*(3 - 4*data)

        def _correction_2(data: np.array):
            """
            Second correction to ideal gas.

            data : np.array
                    data on which to operate
            Returns
            -------
            function_values : np.array
                    result of the operation

            """

            arctan_1 = np.arctan(np.sqrt(4*(data**2) - 1))
            arctan_2 = 8*data*np.arctan((2*data*(4*(data**2) - 3))/(np.sqrt(4*(data**2) - 2)*(4*(data**2) + 1)))
            return 2*data*(3*np.pi - 12*arctan_1 + arctan_2)

        def _piecewise(data: np.array):
            """
            Return a piecewise operation on a set of data
            Parameters
            ----------
            data : np.array
                    data on which to operate

            Returns
            -------
            scaled_data : np.array
                    data that has been operated on.
            """

            # Boundaries on the ideal gsa correction. These go to 73% over half the box size, the most for a cubic box.
            lower_bound = self.parent.box_array[0] / 2
            middle_bound = np.sqrt(2) * self.parent.box_array[0] / 2
            upper_bound = np.sqrt(3) * self.parent.box_array[0] / 2

            # split the data into parts
            split_1 = split_array(data, data <= lower_bound)
            if len(split_1) == 1:
                return _spherical_symmetry(split_1[0])
            else:
                split_2 = split_array(split_1[1], split_1[1] < middle_bound)
                if len(split_2) == 1:
                    return np.concatenate((_spherical_symmetry(split_1[0]), _correction_1(split_2[0])))
                else:
                    return np.concatenate((_spherical_symmetry(split_1[0]),
                                           _correction_1(split_2[0]),
                                           _correction_2(split_2[1])))

        bin_width = self.cutoff / self.number_of_bins
        bin_edges = np.linspace(0.0, self.cutoff, self.number_of_bins)

        return bin_width*_piecewise(np.array(bin_edges))

    def _load_positions(self, indices):
        """
        Load the positions matrix

        This function is here to optimize calculation speed
        """

        return self.parent.load_matrix("Positions", select_slice=np.s_[:, indices], tensor=True)

    def _get_species_names(self, species_tuple):
        """ Get the correct names of the species being studied

        :argument species_tuple (tuple) -- The species tuple i.e (1, 2) corresponding to the rdf being calculated

        :returns names (string) -- Prefix for the saved file
        """

        species = list(self.parent.species)  # load all of the species

        return f"{species[species_tuple[0]]}_{species[species_tuple[1]]}"

    @staticmethod
    def get_neighbour_list(positions, cell=None, batch_size=None):
        """
        Generate the neighbour list

        Parameters
        ----------
        positions: tf.Tensor
            Tensor with shape (number_of_configurations, n_atoms, 3) representing the coordinates
        cell: list
            If periodic boundary conditions are used, please supply the cell dimensions, e.g. [13.97, 13.97, 13.97].
            If the cell is provided minimum image convention will be applied!
        batch_size: int
            Has to be evenly divisible by the the number of configurations.

        Returns
        -------
        generator object which results all distances for the current batch of time steps

        To get the real r_ij matrix for one time_step you can use the following:
            r_ij_mat = np.zeros((n_atoms, n_atoms, 3))
            r_ij_mat[np.triu_indices(n_atoms, k = 1)] = get_neighbour_list(``*args``)
            r_ij_mat -= r_ij_mat.transpose(1, 0, 2)

        """

        def get_triu_indicies(n_atoms):
            """
            Version of np.triu_indices with k=1

            Returns
            ---------
                Returns a vector of size (2, None) instead of a tuple of two values like np.triu_indices
            """
            bool_mat = tf.ones((n_atoms, n_atoms), dtype=tf.bool)
            # Just construct a boolean true matrix the size of one time_step
            indices = tf.where(~tf.linalg.band_part(bool_mat, -1, 0))
            # Get the indices of the lower triangle (without diagonals)
            indices = tf.cast(indices, dtype=tf.int32)  # Get the correct dtype
            return tf.transpose(indices)  # Return the transpose for convenience later

        def get_rij_mat(positions, triu_mask, cell):
            """
            Use the upper triangle of the virtual r_ij matrix constructed of n_atoms * n_atoms matrix and subtract
            the transpose to get all distances once! If PBC are used, apply the minimum image convention.
            """
            r_ij_mat = tf.gather(positions, triu_mask[0], axis=1) - tf.gather(positions, triu_mask[1], axis=1)
            if cell:
                r_ij_mat -= tf.math.rint(r_ij_mat / cell) * cell
            return r_ij_mat

        n_atoms = positions.shape[1]
        triu_mask = get_triu_indicies(n_atoms)

        if batch_size is not None:
            try:
                assert positions.shape[0] % batch_size == 0
            except AssertionError:
                print(
                    f"positions must be evenly divisible by batch_size, but are {positions.shape[0]} and {batch_size}")

            for positions_batch in tf.split(positions, batch_size):
                yield get_rij_mat(positions_batch, triu_mask, cell)
        else:
            yield get_rij_mat(positions, triu_mask, cell)

    @staticmethod
    def _apply_system_cutoff(tensor, cutoff):
        """
        Enforce a cutoff on a tensor
        """

        cutoff_mask = tf.cast(tf.less(tensor, cutoff), dtype=tf.bool)  # Construct the mask

        return tf.boolean_mask(tensor, cutoff_mask)

    @staticmethod
    def _bin_data(distance_tensor, bin_range=None, nbins=500):
        """ Build the histogram number_of_bins for the neighbour lists """

        if bin_range is None:
            bin_range = [0.0, 5.0]

        return tf.histogram_fixed_width(distance_tensor, bin_range, nbins)

    def get_pair_indices(self, len_elements, index_list):
        """
        Get the indicies of the pairs for rdf calculation

        Parameters
        ----------
        len_elements: list
            length of all species/elements in the simulation
        index_list: list
            list of the indices of the species

        Returns
        -------
        array, string: returns a 1D array of the positions of the pairs in the r_ij_mat, name of the pairs

        """
        n_atoms = sum(len_elements)  # Get the total number of atoms in the system
        background = np.full((n_atoms, n_atoms), -1)  # Create a full matrix filled with placeholders
        background[np.triu_indices(n_atoms, k=1)] = np.arange(len(np.triu_indices(n_atoms, k=1)[0]))
        # populate the triu with the respecting indices in the r_ij_matrix
        for tuples in itertools.combinations_with_replacement(index_list, 2):  # Iterate over pairs
            row_slice = (sum(len_elements[:tuples[0]]), sum(len_elements[:tuples[0] + 1]))
            col_slice = (sum(len_elements[:tuples[1]]), sum(len_elements[:tuples[1] + 1]))
            # Yield the slices of the pair being investigated
            names = self._get_species_names(tuples)
            indices = background[slice(*row_slice), slice(*col_slice)]  # Get the indices for the pairs in the r_ij_mat
            yield indices[indices != -1].flatten(), names  # Remove placeholders

    def _calculate_histograms(self):
        """
        Build the rdf dictionary up with histogram data

        Returns
        -------
        update the class state
        """
        for i in tqdm(np.array_split(self.sample_configurations, self.n_batches["Parallel"]), ncols=70):

            positions = self._load_positions(i)  # Load the batch of positions
            positions_tensor = tf.concat(positions, axis=0)  # Combine all elements in one tensor
            positions_tensor = tf.transpose(positions_tensor, (1, 0, 2))  # Change to (time steps, n_atoms, coords)
            # Compute all distance vectors
            r_ij_mat = next(self.get_neighbour_list(positions_tensor, cell=self.parent.box_array))
            for pair, names in self.get_pair_indices([len(x) for x in positions],
                                                     self.index_list):  # Iterate over all pairs
                distance_tensor = tf.norm(tf.gather(r_ij_mat, pair, axis=1), axis=2)  # Compute all distances
                distance_tensor = self._apply_system_cutoff(distance_tensor, self.cutoff)
                # print(names)
                self.rdf[names] += np.array(self._bin_data(distance_tensor, bin_range=self.bin_range, nbins=self.number_of_bins),
                                            dtype=float)

    def _calculate_prefactor(self, species):
        """
        Calculate the relevant prefactor for the analysis

        Parameters
        ----------
        species : str
                The species tuple of the RDF being studied, e.g. Na_Na
        """

        species_scale_factor = 1
        species_split = species.split("_")
        if species_split[0] == species_split[1]:
            species_scale_factor = 2

        # Density of all atoms / total volume
        rho = len(self.parent.species[species_split[1]]['indices']) / self.parent.volume
        ideal_correction = self._get_ideal_gas_probability()  # get the ideal gas value
        numerator = species_scale_factor
        denominator = self.number_of_configurations * rho * ideal_correction * len(self.parent.species[species_split[0]]['indices'])
        prefactor = numerator / denominator

        return prefactor

    def _calculate_radial_distribution_functions(self):
        """
        Take the calculated histograms and apply the correct pre-factor to them to get the correct RDF.
        Returns
        -------
        Updates the class state
        """
        for names in self.key_list:
            prefactor = self._calculate_prefactor(names)  # calculate the prefactor
            self.rdf.update({names: self.rdf.get(names) * prefactor})  # Apply the prefactor
            plt.plot(np.linspace(0.0, self.cutoff, self.number_of_bins), self.rdf.get(names))
            plt.title(names)
            plt.savefig(f'{names}.eps')
            if self.save:  # get the species names
                self._save_data(f'{names}_{self.analysis_name}',
                                [np.linspace(0.0, self.cutoff, self.number_of_bins), self.rdf.get(names)])

        self.parent.radial_distribution_function_state = True  # update the state
        if self.plot:
            self._plot_data()  # Plot the data if necessary

    def run_analysis(self):
        """
        Perform the rdf analysis
        """

        self._collect_machine_properties(scaling_factor=15)  # collect machine properties and determine batch size
        self._calculate_histograms()  # Calculate the RDFs
        self._calculate_radial_distribution_functions()
