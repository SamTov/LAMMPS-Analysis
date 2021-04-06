"""
Atomic transformation to wrap the simulation coordinates.

Summary
-------
When performing analysis on the dynamics of a experiment, it often becomes necessary to reverse the effects of periodic
boundary conditions and track atoms across the box edges. This method uses the box-jump algorithm, whereing particle
positions jumps of more than a half of the box are counted as a crossing of the boundary, to allow the particles to
propagate on into space.
"""

import numpy as np
import os
import tensorflow as tf

from mdsuite.transformations.transformations import Transformations
from mdsuite.database.database import Database
from mdsuite.utils.meta_functions import join_path


class CoordinateWrapper(Transformations):
    """
    Class to unwrap particle coordinates

    Attributes
    ----------
    experiment : object
            The experiment class from which tensor_values will be read and in which it will be saved.

    species : list
            species of atoms to unwrap.

    center_box : bool
            Decision whether or not to center the positions in the box before performing the unwrapping. The default
            value is set to True as this is most common in simulations.

    storage_path : str
            Path to the tensor_values in the database_path, taken directly from the system attribute.

    analysis_name : str
            Name of the analysis, taken directly from the system attribute.

    box_array : list
            Box array of the simulation.

    data : tf.tensor
            Data being unwrapped.

    mask : tf.tensor
            Mask to select and transform crossed tensor_values.
    """

    def __init__(self, experiment: object, species: list = None, center_box: bool = True):
        """
        Standard constructor

        Parameters
        ----------
        experiment : object
                Experiment object to use and update.
        species : list
                List of species to perform unwrapping on
        center_box : bool
                If true, the box coordinates will be centered before the unwrapping occurs
        """

        super().__init__()
        self.experiment = experiment  # assign the experiment attribute
        self.storage_path = self.experiment.storage_path  # get the storage path of the database_path
        self.analysis_name = self.experiment.analysis_name  # get the analysis name
        self.center_box = center_box

        self.box_array = self.experiment.box_array  # re-assign the box array for cleaner code
        self.species = species  # re-assign species
        if species is None:
            self.species = list(self.experiment.species)

        self.data = None  # tensor_values to be unwrapped
        self.mask = None  # image number mask

    def _load_data(self, species):
        """
        Load the tensor_values to be unwrapped
        """
        path = join_path(species, "Unwrapped_Positions")
        self.data = np.array(self.experiment.load_matrix(path=path, select_slice=np.s_[:]))

    def _center_box(self):
        """
        Center the box to match other coordinates.
        Returns
        -------
        adjusts the self.data attribute
        """
        self.data = np.array(self.data)
        self.data[:, :, 0] += (self.box_array[0] / 2)
        self.data[:, :, 1] += (self.box_array[1] / 2)
        self.data[:, :, 2] += (self.box_array[2] / 2)

    def _build_image_mask(self):
        """
        Construct a mask of image numbers
        """

        # Find all distance greater than half a box length and set them to integers.
        self.mask = tf.cast(tf.cast(tf.greater_equal(abs(self.data), np.array(self.box_array) / 2), dtype=tf.int16),
                            dtype=tf.float64)
        self.mask = tf.math.floordiv(x=abs(self.data), y=np.array(self.box_array)/2)

        self.mask = tf.multiply(tf.sign(self.data), self.mask)  # get the correct image sign

    def _apply_mask(self):
        """
        Apply the image mask to the trajectory for the wrapping
        """

        scaled_mask = self.mask * tf.convert_to_tensor(self.box_array, dtype=tf.float64)

        self.data = self.data - scaled_mask  # apply the scaling

    def _save_wrapped_coordinates(self, species, database):
        """
        Save the unwrapped coordinates
        """

        path = join_path(species, 'Positions')
        dataset_structure = {species: {'Positions': tuple(np.shape(self.data))}}
        database.add_dataset(dataset_structure)  # add the dataset to the database_path as resizeable
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2], 'length': len(self.data)}}
        database.add_data(data=self.data,
                          structure=data_structure,
                          start_index=0,
                          batch_size=np.shape(self.data)[1],
                          tensor=True)

    def wrap_particles(self):
        """
        Collect the methods in the class and unwrap the coordinates
        """

        database = Database(name=os.path.join(self.experiment.database_path, "database.hdf5"), architecture='simulation')
        for species in self.species:

            exists = database.check_existence(os.path.join(species, "Positions"))
            # Check if the tensor_values has already been unwrapped
            if exists:
                print(f"Wrapped positions exists for {species}, using the saved coordinates")
            else:
                self._load_data(species)  # load the tensor_values to be unwrapped
                self.data = tf.convert_to_tensor(self.data)
                self._build_image_mask()  # build the image mask
                self._apply_mask()  # Apply the mask and unwrap the coordinates
                if self.center_box:
                    self._center_box()
                self._save_wrapped_coordinates(species, database)  # save the newly unwrapped coordinates

        self.experiment.memory_requirements = database.get_memory_information()
        self.experiment.save_class()  # update the class state

    def run_transformation(self):
        """
        Perform the transformation.
        """
        self.wrap_particles()  # run the transformation
