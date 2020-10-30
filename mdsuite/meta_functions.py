"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: This file contains arbitrary functions used in several different processes. They are often generic and serve
         smaller purposes in order to clean up code in more important parts of the program.
"""

import os
import psutil


def get_dimensionality(box):
    """ Calculate the dimensionality of the system box

    args:
        box (list) -- box array [x, y, z]

    returns:
        dimensions (int) -- dimension of the box i.e, 1 or 2 or 3 (Higher dimensions probably don't make sense just yet)
    """

    if box[0] == 0 or box[1] == 0 or box[2] == 0:
        dimensions = 2

    elif box[0] == 0 and box[1] == 0 or box[0] == 0 and box[2] == 0 or box[1] == 0 and box[2] == 0:
        dimensions = 1

    else:
        dimensions = 3

    return dimensions


def extract_lammps_properties(properties_dict):
    """ Construct generalized property array

    Takes the lammps properties dictionary and constructs and array of properties which can be used by the species
    class.

    agrs:
        properties_dict (dict) -- A dictionary of all the available properties in the trajectory. This dictionary is
        built only from the LAMMPS symbols and therefore must be again processed to extract the useful information.

    returns:
        trajectory_properties (dict) -- A dictionary of the keyword labelled properties in the trajectory. The
        values of the dictionary keys correspond to the array location of the specific piece of data in the set.
    """

    # Define Initial Properties and arrays
    lammps_properties = ["Positions", "Scaled_Positions", "Unwrapped_Positions", "Scaled_Unwrapped_Positions",
                         "Velocities", "Forces", "Box_Images", "Dipole_Orientation_Magnitude",
                         "Angular_Velocity_Spherical",
                         "Angular_Velocity_Non_Spherical", "Torque"]
    trajectory_properties = {}
    system_properties = list(properties_dict)

    if 'x' in system_properties:
        trajectory_properties[lammps_properties[0]] = [properties_dict['x'],
                                                       properties_dict['y'],
                                                       properties_dict['z']]
    if 'xs' in system_properties:
        trajectory_properties[lammps_properties[1]] = [properties_dict['xs'],
                                                       properties_dict['ys'],
                                                       properties_dict['zs']]
    if 'xu' in system_properties:
        trajectory_properties[lammps_properties[2]] = [properties_dict['xu'],
                                                       properties_dict['yu'],
                                                       properties_dict['zu']]
    if 'xsu' in system_properties:
        trajectory_properties[lammps_properties[3]] = [properties_dict['xsu'],
                                                       properties_dict['ysu'],
                                                       properties_dict['zsu']]
    if 'vx' in system_properties:
        trajectory_properties[lammps_properties[4]] = [properties_dict['vx'],
                                                       properties_dict['vy'],
                                                       properties_dict['vz']]
    if 'fx' in system_properties:
        trajectory_properties[lammps_properties[5]] = [properties_dict['fx'],
                                                       properties_dict['fy'],
                                                       properties_dict['fz']]
    if 'ix' in system_properties:
        trajectory_properties[lammps_properties[6]] = [properties_dict['ix'],
                                                       properties_dict['iy'],
                                                       properties_dict['iz']]
    if 'mux' in system_properties:
        trajectory_properties[lammps_properties[7]] = [properties_dict['mux'],
                                                       properties_dict['muy'],
                                                       properties_dict['muz']]
    if 'omegax' in system_properties:
        trajectory_properties[lammps_properties[8]] = [properties_dict['omegax'],
                                                       properties_dict['omegay'],
                                                       properties_dict['omegaz']]
    if 'angmomx' in system_properties:
        trajectory_properties[lammps_properties[9]] = [properties_dict['angmomx'],
                                                       properties_dict['angmomy'],
                                                       properties_dict['angmomz']]
    if 'tqx' in system_properties:
        trajectory_properties[lammps_properties[10]] = [properties_dict['tqx'],
                                                        properties_dict['tqy'],
                                                        properties_dict['tqz']]

    return trajectory_properties


def extract_extxyz_properties(properties_dict):
    """ Construct generalized property array

    Takes the extxyz properties dictionary and constructs and array of properties which can be used by the species
    class.
    """

    # Define Initial Properties and arrays
    extxyz_properties = ['Positions', 'Forces']
    output_properties = []
    system_properties = list(properties_dict)

    if 'pos' in system_properties:
        output_properties.append(extxyz_properties[0])
    if 'force' in system_properties:
        output_properties.append(extxyz_properties[1])

    return output_properties


def line_counter(filename):
    """
    Count the number of lines in a file

    :param filename: (str) name of file to read
    :return: lines: (int) number of lines in the file
    """
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines


def _get_computational_properties(filepath, number_of_configurations):
    """ get the properties of the computer being used """

    file_size = os.path.getsize(filepath)  # Get the size of the file
    available_memory = psutil.virtual_memory().available
    memory_per_configuration = file_size / number_of_configurations  # get the memory per configuration
    database_memory = 0.1 * available_memory  # We take 10% of the available memory
    initial_batch_number = int(database_memory / memory_per_configuration)  # trivial batch allocation

    return initial_batch_number, database_memory, file_size


def optimize_batch_size(filepath, number_of_configurations):
    """ Optimize the size of batches during initial processing

    During the database construction a batch size must be chosen in order to process the trajectories with the
    least RAM but reasonable performance.
    """

    computer_statistics = _get_computational_properties(filepath, number_of_configurations)  # Get computer statistics

    batch_number = None  # Instantiate parameter for correct syntax

    if computer_statistics[2] < computer_statistics[1]:
        batch_number = number_of_configurations
    else:
        remainder = 1000000000
        for i in range(10):
            r_temp = number_of_configurations % (computer_statistics[0] - i)
            if r_temp <= remainder:
                batch_number = computer_statistics[0] - i

    if batch_number > 1000:
        batch_number = 1000

    return batch_number


def linear_fitting_function(x, a, b):
    """ Linear function for line fitting

    In many cases, namely those involving an Einstein relation, a linear curve must be fit to some data. This function
    is called by the scipy curve_fit module as the model to fit to.

    args:
        x (list) -- x data for fitting
        a (float) -- fitting parameter of the gradient
        b (float) -- fitting parameter for the y intercept
    """
    return a * x + b
