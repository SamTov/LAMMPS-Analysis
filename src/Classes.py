"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Class functionality of the program
"""

import numpy as np
import os
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import Methods
import pickle
import h5py as hf
from alive_progress import alive_bar
import time

class Trajectory(Methods.Trajectory_Methods):
    """ Trajectory from simulation

    Attributes:

    Methods:

    """

    def __init__(self, analysis_name):
        """ Initialise with filename """

        self.filename = None
        self.analysis_name = analysis_name
        self.temperature = None
        self.volume = None
        self.species = None
        self.number_of_atoms = None
        self.properties = None
        self.property_groups = None
        self.dimensions = None
        self.box_array = None
        self.number_of_configurations = None
        self.singular_diffusion_coefficients = None
        self.distinct_diffusion_coefficients = None
        self.ionic_conductivity = None

    def From_User(self):
        """ Get system specific inputs

        If a new project is called, this function will gather extra data from the user
        """

        self.filename = input("File name: ")
        self.temperature = float(input("Temperature: "))

    def Save_Class(self):
        """ Saves class instance

        In order to keep properties of a class the state must be stored. This method will store the instance of the
        class for later re-loading
        """

        save_file = open("{0}.txt".format(self.analysis_name), 'wb')
        save_file.write(pickle.dumps(self.__dict__))
        save_file.close()

    def Load_Class(self):
        """ Load class instance

        A function to load a class instance given the project name.
        """
        os.chdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))

        class_file = open('{0}.txt'.format(self.analysis_name), 'rb')
        pickle_data = class_file.read()
        class_file.close()

        self.__dict__ = pickle.loads(pickle_data)

        os.chdir('../../src')

    def Process_Input_File(self):
        """ Process the input file

        A trivial function to get the format of the input file. Will probably become more useful when we add support
        for more file formats.
        """

        if self.filename[-6:] == 'extxyz':
            file_format = 'extxyz'
        else:
            file_format = 'lammps'

        return file_format

    def Get_System_Properties(self, file_format):
        """ Get the properties of the system

        This method will call the Get_X_Properties depending on the file format. This function will update all of the
        class attributes and is necessary for the operation of the Build database method.

        args:
            file_format (str) -- Format of the file being read
        """

        if file_format == 'lammps':
            Methods.Trajectory_Methods.Get_LAMMPS_Properties(self)
        else:
            Methods.Trajectory_Methods.Get_EXTXYZ_Properties(self)

    def Build_Database(self):
        """ Build the 'database' for the analysis

        A method to build the database in the hdf5 format. Within this method, several other are called to develop the
        database skeleton, get configurations, and process and store the configurations. The method is accompanied
        by a loading bar which should be customized to make it more interesting.
        """

        self.From_User() # Get additional information

        # Create new analysis directory and change into it
        os.mkdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))
        os.chdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))

        file_format = self.Process_Input_File()  # Collect data array
        self.Get_System_Properties(file_format) # Update class attributes
        Methods.Trajectory_Methods.Build_Database_Skeleton(self)

        print("Beginning Build database")

        with hf.File("{0}.hdf5".format(self.analysis_name), "r+") as database:
            with open(self.filename) as f:
                counter = 0
                with alive_bar(int(self.number_of_configurations/10), bar = 'blocks', spinner = 'dots') as bar:
                    for i in range(int(self.number_of_configurations / 10)):
                        test = Methods.Trajectory_Methods.Read_Configurations(self, 10, f)

                        Methods.Trajectory_Methods.Process_Configurations(self, test, database, counter)

                        counter += 10
                        bar()

        self.Save_Class()
        os.chdir('../')

        print("\n ** Database has been constructed and saved for {0} ** \n".format(self.analysis_name))

    def Unwrap_Coordinates(self):
        """ Unwrap coordinates of trajectory

        For a number of properties the input data must in the form of unwrapped coordinates. This function takes the
        stored trajectory and returns the unwrapped coordinates so that they may be used for analysis.
        """
        os.chdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))  # Change to correct directory

        box_array = self.box_array # Get the static box array --  NEED A NEW METHOD FOR V NEQ CONST SYSTEMS E.G NPT

        def Center_Box(positions_matrix):
            """ Center atoms in box

            A function to center the coordinates in the box so that the unwrapping is performed equally in all
            directions.

            args:
                positions_matrix (array) -- array of positions to be unwrapped.
            """

            positions_matrix -= (box_array[0] / 2)

        def Unwrap(database):
            """ Unwrap the coordinates

            Central function in the unwrapping method. This function will detect jumps across the box boundary and
            shifts the position of the atoms accordingly.
            """

            print("\n --- Beginning to unwrap coordinates --- \n")

            for item in self.species:
                # Construct the positions matrix -- Only temporary, we should make this memory safe
                positions_matrix = np.dstack((database[item]["Positions"]['x'],
                                             database[item]["Positions"]['y'],
                                             database[item]["Positions"]['z']))

                Center_Box(positions_matrix) # Center the box at (0, 0, 0)

                for j in range(len(positions_matrix)):
                    difference = np.diff(positions_matrix[j], axis=0)  # Difference between all atoms in the array

                    # Indices where the atoms jump in the original array
                    box_jump = [np.where(abs(difference[:, 0]) >= (box_array[0] / 2))[0],
                                np.where(abs(difference[:, 1]) >= (box_array[1] / 2))[0],
                                np.where(abs(difference[:, 2]) >= (box_array[2] / 2))[0]]

                    # Indices of first box cross
                    box_cross = [box_jump[0] + 1, box_jump[1] + 1, box_jump[2] + 1]
                    box_cross[0] = box_cross[0]
                    box_cross[1] = box_cross[1]
                    box_cross[2] = box_cross[2]

                    for k in range(len(box_cross[0])):
                        positions_matrix[j][:, 0][box_cross[0][k]:] -= np.sign(difference[box_cross[0][k] - 1][0]) * \
                                                                          box_array[0]
                    for k in range(len(box_cross[1])):
                        positions_matrix[j][:, 1][box_cross[1][k]:] -= np.sign(difference[box_cross[1][k] - 1][1]) * \
                                                                          box_array[1]
                    for k in range(len(box_cross[2])):
                        positions_matrix[j][:, 2][box_cross[2][k]:] -= np.sign(difference[box_cross[2][k] - 1][2]) * \
                                                                          box_array[2]

                print(np.array([positions_matrix[i][:, 2] for i in range(len(positions_matrix))]))

                database[item].create_group("Unwrapped_Positions")
                database[item]["Unwrapped_Positions"].create_dataset('x',
                                                                     data = np.array([positions_matrix[i][:, 0] for i
                                                                                      in range(len(positions_matrix))]))
                database[item]["Unwrapped_Positions"].create_dataset('y',
                                                                     data = np.array([positions_matrix[i][:, 1] for i
                                                                                      in range(len(positions_matrix))]))
                database[item]["Unwrapped_Positions"].create_dataset('z',
                                                                     data = np.array([positions_matrix[i][:, 2] for i
                                                                                      in range(len(positions_matrix))]))



        with hf.File("{0}.hdf5".format(self.analysis_name), "r+") as database:
            Unwrap(database)

        print("\n --- Finished unwrapping coordinates --- \n")
        os.chdir('../')

    def Einstein_Diffusion_Coefficients(self):
        """ Calculate the Einstein self diffusion coefficients

            A function to implement the Einstein method for the calculation of the self diffusion coefficients
            of a liquid. In this method, unwrapped trajectories are read in and the MSD of the positions calculated and
            a gradient w.r.t time is calculated over several ranges to calculate an error measure.

            Data is loaded from the working directory.
        """

        os.chdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))  # Change into analysis directory
        database = hf.File("{0}.hdf5".format(self.analysis_name), "r")

        def fitting_function(x, a, b):
            """ Function for use in fitting """
            return a*x + b

        positions_matrix = []
        for item in self.species:
            positions_matrix.append(np.dstack((database[item]["Unwrapped_Positions"]['x'],
                                             database[item]["Unwrapped_Positions"]['y'],
                                             database[item]["Unwrapped_Positions"]['z'])))

        print(positions_matrix[0][0])

        def Singular_Diffusion_Coefficients():
            """ Calculate singular diffusion coefficients

            Implement the Einstein method for the calculation of the singular diffusion coefficient. This is performed
            using unwrapped coordinates, generated by the unwrap method. From these values, the mean square displacement
            of each atom is calculated and averaged over all the atoms in the system.
            """

            diffusion_coefficients = {}
            for i in range(len(positions_matrix)): # Loop over species
                msd_x = []
                msd_y = []
                msd_z = []
                for j in range(len(positions_matrix[i])): # Loop over number of atoms of species i
                    msd_x.append((positions_matrix[i][j][:, 0] - positions_matrix[i][j][0][0])**2)
                    msd_y.append((positions_matrix[i][j][:, 1] - positions_matrix[i][j][0][1])**2)
                    msd_z.append((positions_matrix[i][j][:, 2] - positions_matrix[i][j][0][2])**2)

                # Take averages
                msd_x = np.mean(msd_x, axis=0)
                msd_y = np.mean(msd_y, axis=0)
                msd_z = np.mean(msd_z, axis=0)

                msd = (msd_x + msd_y + msd_z) # Calculate the total MSD

                # Perform unit conversions
                msd = msd*(1E-20)
                #time = 100*np.array([i for i in range(len(msd))])*(1E-12)*(0.002) # Need to solve this time problem.
                time = np.linspace(0.0, 19151, len(msd))*(1E-12)
                
                np.save('time.npy', time)
                np.save('{0}.npy'.format(i), msd)
                popt, pcov = curve_fit(fitting_function, time, msd)
                diffusion_coefficients[list(self.species)[i]] = popt[0]/6

                #plt.loglog(time, fitting_function(time, *popt))
                plt.loglog(time, msd)
                plt.show()
                plt.plot(time, msd)
                plt.show()

            return diffusion_coefficients

        def Distinct_Diffusion_Coefficients():
            """ Calculate the Distinct Diffusion Coefficients

            Use the Einstein method to calculate the distinct diffusion coefficients of the system from the mean
            square displacement, as calculated from different atoms. This value is averaged over all the possible
            combinations of atoms for the best fit.
            """

            distinct_diffusion_coefficients = {}

            msd_x = np.array([0.0 for i in range(len(positions_matrix[0][0]))])
            msd_y = np.array([0.0 for i in range(len(positions_matrix[0][0]))])
            msd_z = np.array([0.0 for i in range(len(positions_matrix[0][0]))])

            for i in range(len(positions_matrix[0])):
                for j in range(len(positions_matrix[0])):
                    if j == i:
                        continue

                    msd_x += (positions_matrix[0][i][:, 0] - positions_matrix[0][i][0][0])*\
                            (positions_matrix[1][j][:, 0] - positions_matrix[1][j][0][0])
                    msd_y += (positions_matrix[0][i][:, 1] - positions_matrix[0][i][0][1]) * \
                            (positions_matrix[1][j][:, 1] - positions_matrix[1][j][0][1])
                    msd_z += (positions_matrix[0][i][:, 2] - positions_matrix[0][i][0][2]) * \
                            (positions_matrix[1][j][:, 2] - positions_matrix[1][j][0][2])

            msd_x = msd_x
            msd_y = msd_y
            msd_z = msd_z

            msd = (1E-20)*(len(positions_matrix[1]) + len(positions_matrix[0]))*(msd_x + msd_y + msd_z)/(len(positions_matrix[0])*(len(positions_matrix[0])-1))
            time = np.linspace(0.0, 15863.2, len(msd)) * (1E-12)

            plt.plot(time, msd)
            plt.show()

        singular_diffusion_coefficients = Singular_Diffusion_Coefficients()
        #Distinct_Diffusion_Coefficients()

        print(singular_diffusion_coefficients)
        os.chdir('../../src'.format(self.analysis_name))

    def Green_Kubo_Diffusion_Coefficients(self):
        """ Calculate the Green_Kubo Diffusion coefficients

        Function to implement a Green-Kubo method for the calculation of diffusion coefficients whereby the velocity
        autocorrelation function is integrated over and divided by 3. Autocorrelation is performed using the scipy
        fft correlate function in order to speed up the calculation.
        """

        os.chdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))  # Change to correct directory
        database = hf.File("{0}.hdf5".format(self.analysis_name), "r")

        velocity_matrix = []
        for item in self.species:
            velocity_matrix.append(np.dstack((database[item]["Velocities"]['x'],
                                             database[item]["Velocities"]['y'],
                                             database[item]["Velocities"]['z'])))

        def Singular_Diffusion_Coefficients():
            vacf_a = np.zeros(2 * len(velocity_matrix[0][0]) - 1)
            vacf_b = np.zeros(2 * len(velocity_matrix[0][0]) - 1)

            for i in range(len(velocity_matrix[0])):
                vacf_a += (1 / len(velocity_matrix[0])) * np.array(
                    signal.correlate(velocity_matrix[0][i][:, 0], velocity_matrix[0][i][:, 0], mode='full',
                                     method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 1], velocity_matrix[0][i][:, 1], mode='full',
                                     method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 2], velocity_matrix[0][i][:, 2], mode='full',
                                     method='fft'))
                vacf_b += (1 / len(velocity_matrix[1])) * np.array(
                    signal.correlate(velocity_matrix[1][i][:, 0], velocity_matrix[1][i][:, 0], mode='full',
                                     method='fft') +
                    signal.correlate(velocity_matrix[1][i][:, 1], velocity_matrix[1][i][:, 1], mode='full',
                                     method='fft') +
                    signal.correlate(velocity_matrix[1][i][:, 2], velocity_matrix[1][i][:, 2], mode='full',
                                     method='fft'))

            sub_a = vacf_a
            sub_b = vacf_b
            vacf_a = (1 / (2*len(vacf_a) - 1)) * vacf_a[int(len(vacf_a) / 2):] * ((1E-20) / (1E-24))
            vacf_b = (1 / (2*len(vacf_b)-1)) * vacf_b[int(len(vacf_b) / 2):] * ((1E-20) / (1E-24))

            time = np.linspace(0.0, 19151, len(vacf_a)) * (1E-12)

            D_a = np.trapz(vacf_a, x=time) / 3
            D_b = np.trapz(vacf_b, x=time) / 3
            print(D_a)
            print(D_b)

            plt.plot(time, vacf_a)
            plt.plot(time, vacf_b)
            plt.show()

            return D_a, D_b, sub_a, sub_b

        def Distinct_Diffusion_Coefficients():

            vacf_a = np.zeros(2 * len(velocity_matrix[0][0]) - 1)
            vacf_b = np.zeros(2 * len(velocity_matrix[0][0]) - 1)
            vacf_c = np.zeros(2 * len(velocity_matrix[0][0]) - 1)

            velocity_a = []
            velocity_b = []
            for i in range(len(velocity_matrix[0][0])):
                velocity_a.append(np.sum(velocity_matrix[0][:, i], axis=0))
                velocity_b.append(np.sum(velocity_matrix[1][:, i], axis=0))

            velocity_a = np.array(velocity_a)
            velocity_b = np.array(velocity_b)

            for i in range(len(velocity_matrix[0])):
                vacf_a += (1 / (499 * 498)) * np.array(
                    signal.correlate(velocity_matrix[0][i][:, 0], velocity_a[:, 0] - velocity_matrix[0][i][0][0],
                                     mode='full', method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 0], velocity_a[:, 1] - velocity_matrix[0][i][1][0],
                                     mode='full', method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 0], velocity_a[:, 2] - velocity_matrix[0][i][2][0],
                                     mode='full', method='fft'))
                vacf_b += (1 / (499 * 498)) * np.array(
                    signal.correlate(velocity_matrix[1][i][:, 0], velocity_b[:, 0] - velocity_matrix[1][i][0][0],
                                     mode='full', method='fft') +
                    signal.correlate(velocity_matrix[1][i][:, 0], velocity_b[:, 1] - velocity_matrix[1][i][1][0],
                                     mode='full', method='fft') +
                    signal.correlate(velocity_matrix[1][i][:, 0], velocity_b[:, 2] - velocity_matrix[1][i][2][0],
                                     mode='full', method='fft'))
                vacf_c += (1 / (500 * 500)) * np.array(
                    signal.correlate(velocity_b[:, 0], velocity_b[:, 0], mode='full', method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 0], velocity_b[:, 1], mode='full', method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 0], velocity_b[:, 2], mode='full', method='fft'))

            vacf_a = self.number_of_atoms * (1 / (len(vacf_a))) * vacf_a[int(len(vacf_a) / 2):] * (1E-20) / (1E-24)
            vacf_b = self.number_of_atoms * (1 / (len(vacf_b))) * vacf_b[int(len(vacf_b) / 2):] * (1E-20) / (1E-24)
            vacf_c = self.number_of_atoms * (1 / (len(vacf_c))) * vacf_c[int(len(vacf_c) / 2):] * (1E-20) / (1E-24)

            D_a = np.trapz(vacf_a, x=time) / 6
            D_b = np.trapz(vacf_b, x=time) / 6
            D_c = np.trapz(vacf_c, x=time) / 6

            print(D_a)
            print(D_b)
            print(D_c)

            plt.plot(time, vacf_a)
            plt.plot(time, vacf_b)
            plt.plot(time, vacf_c)
            plt.show()

        _, _, _, _ = Singular_Diffusion_Coefficients()
        #Distinct_Diffusion_Coefficients()
        os.chdir('../../src')  # Change to correct directory

    def Nernst_Einstein_Conductivity(self):
        """ Calculate Nernst-Einstein Conductivity

        A function to determine the Nernst-Einstein as well as the corrected Nernst-Einstein
        conductivity of a system.
        """
        pass

    def Einstein_Helfand_Conductivity(self):
        """ Calculate the Einstein-Helfand Conductivity

        A function to use the mean square displacement of the dipole moment of a system to extract the
        ionic conductivity
        """


        def func(x, a, b):
            return a*x + b

        q = 1.60217662E-19
        kb = 1.38064852E-23 # Define the Boltzmann constant

        measurement_range = 70000 # 10ns

        os.chdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))
        database = hf.File("{0}.hdf5".format(self.analysis_name), "r")

        position_matrix = []
        for item in self.species:
            position_matrix.append(np.dstack((database[item]["Unwrapped_Positions"]['x'],
                                             database[item]["Unwrapped_Positions"]['y'],
                                             database[item]["Unwrapped_Positions"]['z'])))

        positions_a = []
        positions_b= []
        for i in range(len(position_matrix[0][0])):
            positions_a.append(np.sum(position_matrix[0][:, i], axis=0))
            positions_b.append(np.sum(position_matrix[1][:, i], axis=0))

        dipole_moment = q * (np.array(positions_a) - np.array(positions_b))

        dipole_moment_msd_x = np.array([0.0 for i in range(measurement_range)])
        dipole_moment_msd_y = np.array([0.0 for i in range(measurement_range)])
        dipole_moment_msd_z = np.array([0.0 for i in range(measurement_range)])

        for i in range(0, len(position_matrix[0][0]) - (measurement_range-1)):
            dipole_moment_msd_x += dipole_moment[i:i+measurement_range, 0] - dipole_moment[i][0]
            dipole_moment_msd_y += dipole_moment[i:i+measurement_range, 1] - dipole_moment[i][1]
            dipole_moment_msd_z += dipole_moment[i:i+measurement_range, 2] - dipole_moment[i][2]

        dipole_msd = (1/measurement_range)*np.array(dipole_moment_msd_x**2 + dipole_moment_msd_y**2 + dipole_moment_msd_z**2)*(1E-20)

        time = np.linspace(0.0, 14000, len(dipole_msd)) * (1E-12)

        popt, pcov = curve_fit(func, time[25000:], dipole_msd[25000:])

        prefactor = (1/500)*(1/(6*self.temperature*(self.volume*1E-30)*kb))
        sigma = popt[0]*prefactor

        print(sigma/100)

        plt.plot(time, dipole_msd)
        plt.plot(time, func(time, *popt))
        plt.show()
        plt.loglog(time, dipole_msd)
        plt.show()

        os.chdir('../../src'.format(self.analysis_name))


    def Green_Kubo_Conductivity(self):
        """ Calculate Green-Kubo Conductivity

        A function to use the current autocorrelation function to calculate the Green-Kubo ionic conductivity of the
        system being studied.
        """

        q = 1.60217662E-19 # Define elementary charge
        kb = 1.38064852E-23 # Define the Boltzmann constant
        os.chdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))  # Change to correct directory
        database = hf.File("{0}.hdf5".format(self.analysis_name), "r")

        velocity_matrix = []
        for item in self.species:
            velocity_matrix.append(np.dstack((database[item]["Velocities"]['x'],
                                             database[item]["Velocities"]['y'],
                                             database[item]["Velocities"]['z'])))

        vacf_a = np.zeros(2 * len(velocity_matrix[0][0]) - 1)
        vacf_b = np.zeros(2 * len(velocity_matrix[0][0]) - 1)
        vacf_c = np.zeros(2 * len(velocity_matrix[0][0]) - 1)

        velocity_a = []
        velocity_b = []
        for i in range(len(velocity_matrix[0][0])):
            velocity_a.append(np.sum(velocity_matrix[0][:, i], axis=0))
            velocity_b.append(np.sum(velocity_matrix[1][:, i], axis=0))

        current = q * (np.array(velocity_a) - np.array(velocity_b))

        jacf = (signal.correlate(current[:, 0], current[:, 0], mode='full', method='fft') +
                signal.correlate(current[:, 1], current[:, 1], mode='full', method='fft') +
                signal.correlate(current[:, 2], current[:, 2], mode='full', method='fft'))

        jacf = (1 / (2*len(jacf) - 1)) * (jacf[int((len(jacf) / 2)):]) * ((1E-20) / (1E-24))

        time = np.linspace(0.0, 15863.2, len(jacf)) * (1E-12)
        sigma = (1/(3 * self.temperature * ((self.volume*1E-30) * kb))) * np.trapz(jacf, x=time)
        print(sigma/100)

        plt.plot(time, jacf)
        plt.show()

        os.chdir('../../src'.format(self.analysis_name))
