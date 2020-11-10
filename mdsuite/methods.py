"""
Author: Samuel Tovey ; Francisco Torres
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Larger methods used in the Experiment class
"""
import pickle

import h5py as hf
import mendeleev
import numpy as np


import mdsuite.constants as Constants
import mdsuite.meta_functions as Meta_Functions


class ProjectMethods:
    """ methods to be used in the Experiment class """

    def get_lammps_properties(self):
        """ Get the properties of the system from a custom lammps dump file

            returns:
                species_summary (dict) -- Dictionary containing all the species in the systems
                                          and how many of them there are in each configuration.
                properties_summary (dict) -- All the properties available in the dump file for
                                             analysis and their index in the file
        """

        # Define necessary properties and attributes
        species_summary = {}
        properties_summary = {}
        lammps_properties_labels = {'x', 'y', 'z',
                                    'xs', 'ys', 'zs',
                                    'xu', 'yu', 'zu',
                                    'xsu', 'ysu', 'zsu',
                                    'ix', 'iy', 'iz',
                                    'vx', 'vy', 'vz',
                                    'fx', 'fy', 'fz',
                                    'mux', 'muy', 'muz', 'mu',
                                    'omegax', 'omegay', 'omegaz',
                                    'angmomx', 'angmomy', 'angmomz',
                                    'tqx', 'tqy', 'tqz'}

        nlines_header_block = 9
        with open(self.filename) as f:
            head = [next(f).split() for i in range(nlines_header_block)]
            f.seek(0)  # Go back to the start of the file
            # Calculate the number of atoms and configurations in the system
            number_of_atoms = int(head[3][0])
            # Get first configuration
            data_array = [next(f).split() for i in range(number_of_atoms + nlines_header_block)]  # Get first configuration
            second_configuration = [next(f).split() for i in range(number_of_atoms + nlines_header_block)] # Get the second

        number_of_lines = Meta_Functions.line_counter(self.filename)
        number_of_configurations = int(number_of_lines / (number_of_atoms + nlines_header_block)) # n of timesteps
        batch_size = Meta_Functions.optimize_batch_size(self.filename, number_of_configurations)

        time_0 = float(data_array[1][0])
        time_1 = float(second_configuration[1][0])
        sample_rate = time_1 - time_0
        time_N = (number_of_configurations - number_of_configurations % batch_size)*sample_rate

        # Get the position of the element keyword so that any format can be given
        for i in range(len(data_array[8])):
            if data_array[8][i] == "element":
                element_index = i - 2

        # Find the information regarding species in the system and construct a dictionary
        for i in range(9, number_of_atoms + 9):
            if data_array[i][element_index] not in species_summary:
                species_summary[data_array[i][element_index]] = {}
                species_summary[data_array[i][element_index]]['indices'] = []

            species_summary[data_array[i][element_index]]['indices'].append(i)

        # Find properties available for analysis
        for i in range(len(data_array[8])):
            if data_array[8][i] in lammps_properties_labels:
                properties_summary[data_array[8][i]] = i - 2

        # Get the box size from the system
        box = [(float(data_array[5][1][:-10]) - float(data_array[5][0][:-10])) * 10,
               (float(data_array[6][1][:-10]) - float(data_array[6][0][:-10])) * 10,
               (float(data_array[7][1][:-10]) - float(data_array[7][0][:-10])) * 10]

        # Update class attributes with calculated data
        self.batch_size = batch_size
        self.dimensions = Meta_Functions.get_dimensionality(box)
        self.box_array = box
        self.volume = box[0] * box[1] * box[2]
        self.species = species_summary
        self.number_of_atoms = number_of_atoms
        self.properties = properties_summary
        self.number_of_configurations = number_of_configurations
        self.time_dimensions = [0.0, time_N*self.time_step*self.time_unit]
        self.sample_rate = sample_rate

    def get_extxyz_properties(self, data_array):
        """ Function to process extxyz input files """

        print("This functionality does not currently work")
        return

    def build_species_dictionary(self):
        """ Add information to the species dictionary

        A fundamental part of this package is species specific analysis. Therefore, the mendeleev python package is
        used to add important species specific information to the class. This will include the charge of the ions which
        will be used in conductivity calculations.

        returns:
            This method will update the class attributes in place and therefore, will not return anything explicitly.
        """

        for element in self.species:
            try:
                temp = mendeleev.element(element)
            except:
                self.species[element]['charge'] = [0]
                continue

            charge = [] # Define empty charge array
            for ir in temp.ionic_radii:
                if ir.most_reliable is not True:
                    continue
                else:
                    charge.append(ir.charge)

            if not temp.ionic_radii:
                self.species[element]['charge'] = [0]
            elif len(charge) == 0:
                self.species[element]['charge'] = [temp.ionic_radii[0].charge] # Case where most_reliable is all False
            elif all(elem == charge[0] for elem in charge) is True:
                self.species[element]['charge'] = [charge[0]]
            else:
                self.species[element]['charge'] = [charge]

            mass = []
            for iso in temp.isotopes:
                mass.append(iso.mass)
            self.species[element]['mass'] = mass

    def _build_database_skeleton(self):
        """ Build skeleton of the hdf5 database

        Gathers all of the properties of the system using the relevant functions. Following the gathering
        of the system properties, this function will read through the first configuration of the dataset, and
        generate the necessary database structure to allow for the following generation to take place. This will
        include the separation of species, atoms, and properties. For a full description of the data structure,
        look into the documentation.
        """

        database = hf.File('{0}/{1}/{1}.hdf5'.format(self.filepath, self.analysis_name), 'w', libver='latest')

        property_groups = Meta_Functions.extract_lammps_properties(self.properties)  # Get the property groups
        self.property_groups = property_groups

        # Build the database structure
        for item in self.species:
            database.create_group(item)
            for property in property_groups:
                database[item].create_group(property)
                database[item][property].create_dataset("x", (len(self.species[item]['indices']), self.number_of_configurations-
                                                              self.number_of_configurations % self.batch_size),
                                                        compression="gzip", compression_opts=9)
                database[item][property].create_dataset("y", (len(self.species[item]['indices']), self.number_of_configurations-
                                                              self.number_of_configurations % self.batch_size),
                                                        compression="gzip", compression_opts=9)
                database[item][property].create_dataset("z", (len(self.species[item]['indices']), self.number_of_configurations -
                                                              self.number_of_configurations % self.batch_size),
                                                        compression="gzip", compression_opts=9)

    def read_configurations(self, N, f):
        """ Read in N configurations

        This function will read in N configurations from the file that has been opened previously by the parent method.

        args:

            N (int) -- Number of configurations to read in. This will depend on memory availability and the size of each
                        configuration. Automatic setting of this variable is not yet available and therefore, it will be set
                        manually.
            f (obj) --
        """

        data = []

        for i in range(N):
            # Skip header lines
            for j in range(9):
                f.readline()

            for k in range(self.number_of_atoms):
                data.append(f.readline().split())

        return np.array(data)

    def process_configurations(self, data, database, counter):
        """ Process the available data

        Called during the main database creation. This function will calculate the number of configurations within the
        raw data and process it.

        args:
            data (numpy array) -- Array of the raw data for N configurations.
            database (object) --
            counter (int) --
        """

        # Re-calculate the number of available configurations for analysis
        partitioned_configurations = int(len(data) / self.number_of_atoms)

        for item in self.species:
            # get the new indices for the positions
            positions = np.array([np.array(self.species[item]['indices']) + i * self.number_of_atoms - 9 for i in
                                  range(int(partitioned_configurations))]).flatten()
            # Fill the database
            for property_group in self.property_groups:
                database[item][property_group]["x"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.property_groups[property_group][0]].astype(float).reshape(
                        (len(self.species[item]['indices']), partitioned_configurations), order='F')

                database[item][property_group]["y"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.property_groups[property_group][1]].astype(float).reshape(
                        (len(self.species[item]['indices']), partitioned_configurations), order='F')

                database[item][property_group]["z"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.property_groups[property_group][2]].astype(float).reshape(
                        (len(self.species[item]['indices']), partitioned_configurations), order='F')

    def print_data_structrure(self):
        """ Print the data structure of the hdf5 dataset """

        database = hf.File("{0}/{1}/{1}.hdf5".format(self.filepath, self.analysis_name), "r")

    def write_xyz(self, property="Positions", species=None):
        """ Write an xyz file from database array

        For some of the properties calculated it is beneficial to have an xyz file for analysis with other platforms.
        This function will write an xyz file from a numpy array of some property. Can be used in the visualization of
        trajectories.

        kwargs:
            property (str) -- Which property would you like to print
            species (list) -- List of species for which you would like to write the file
        """

        if species == None:
            species = list(self.species.keys())

        data_matrix = self.load_matrix(property, species)

        with open(f"{self.filepath}/{self.analysis_name}/{property}_{'_'.join(species)}.xyz", 'w') as f:
            for i in range(self.number_of_configurations):
                f.write(f"{self.number_of_atoms}\n")
                f.write("Generated by the mdsuite xyz writer\n")
                for j in range(len(species)):
                    for atom in data_matrix[j]:
                        f.write(f"{species[j]:<2}    {atom[i][0]:>9.4f}    {atom[i][1]:>9.4f}    {atom[i][2]:>9.4f}\n")

    def save_class(self):
        """ Saves class instance

        In order to keep properties of a class the state must be stored. This method will store the instance of the
        class for later re-loading
        """

        save_file = open("{0}/{1}/{1}.bin".format(self.filepath, self.analysis_name), 'wb')
        save_file.write(pickle.dumps(self.__dict__))
        save_file.close()

    def load_class(self):
        """ Load class instance

        A function to load a class instance given the project name.
        """

        class_file = open('{0}/{1}/{1}.bin'.format(self.filepath, self.analysis_name), 'rb')
        pickle_data = class_file.read()
        class_file.close()

        self.__dict__ = pickle.loads(pickle_data)

    def print_class_attributes(self):
        """ Print all attributes of the class """

        attributes = []
        for item in vars(self).items():
            attributes.append(item)
        for tuple in attributes:
            print(f"{tuple[0]}: {tuple[1]}")

        return attributes

    @staticmethod
    def units_to_si(units_system, dimension):
        """ Passes the given dimension to SI units.

        It is easier to work in SI units always, to avoid mistakes.

        Parameters
        ----------
        units_system (str) -- current unit system
        dimension (str) -- dimension you would like to change

        Returns
        -------
        conv_factor (float) -- conversion factor to pass to SI

        Examples
        --------
        Pass from metal units of time (ps) to SI

        >>> units_to_si('metal', 'time')
        1e-12
        """
        units = {
            "metal": {'time': 1e-12, 'length': 1e-10, 'energy': 1.6022e-19},
            "real": {'time': 1e-15, 'length': 1e-10, 'energy': 4184 / Constants.avogadro_constant},
        }

        return units[units_system][dimension]
