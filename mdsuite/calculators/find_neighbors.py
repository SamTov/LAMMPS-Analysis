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
MDSuite module for the computation of the number of atoms bonded to each atom.
This can be useful to compute coordination fractions for carbon for example.
"""
from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Union

import numpy as np
import tensorflow as tf
from bokeh.models import HoverTool
from bokeh.palettes import Category10  # select a palette
from bokeh.plotting import figure
from scipy.spatial import KDTree
from tqdm import tqdm

from mdsuite.calculators.calculator import call
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties

# Import user packages
from mdsuite.utils.meta_functions import join_path

log = logging.getLogger(__name__)


@dataclass
class Args:
    """
    Data class for the saved properties.
    """

    number_of_configurations: int
    data_range: int
    max_bond_length: float
    start: int
    stop: int
    molecules: bool
    species: list
    correlation_time: int
    atom_selection: np.s_


class FindNeighbors(TrajectoryCalculator, ABC):
    """
    Class for the calculation of the radial distribution function

    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    data_range :
            Number of configurations to use in each ensemble
    x_label : str
            X label of the tensor_values when plotted
    y_label : str
            Y label of the tensor_values when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database_path for the analysis

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------

    .. code-block:: python

        project = mdsuite.Project()
        project.run.FindNeighbors(number_of_configurations=500, max_bond_length)

    """

    def __init__(self, **kwargs):
        """
        Constructor for the RDF calculator.

        Attributes
        ----------
        kwargs: see RunComputation class for all the passed arguments
        """
        super().__init__(**kwargs)

        self.scale_function = {
            "quadratic": {"outer_scale_factor": 10, "inner_scale_factor": 5}
        }
        self.loaded_property = mdsuite_properties.positions
        self.x_label = r"$$Timestep$$"
        self.y_label = r"$$Atoms$$"
        self.analysis_name = "Find_Neighbors"
        self.result_series_keys = ["num. neighbors", "counts"]

        self._dtype = tf.float32
        self.use_tf_function = None
        self.override_n_batches = None
        self.index_list = None
        self.sample_configurations = None
        self.key_list = None
        self.rdf = None

    @call
    def __call__(
        self,
        plot: bool = False,
        max_bond_length: float = None,
        save: bool = True,
        start: int = 0,
        stop: int = None,
        number_of_configurations: int = 1,
        # TODO: this is unclear to me. related to Funtion corerct_batch_properties
        atom_selection: Union[np.s_, dict] = np.s_[:],
        species: list = None,
        molecules: bool = False,
        **kwargs,
    ):
        """
        Compute the RDF with the given user parameters

        Parameters
        ----------
        plot: bool
            Plot the RDF after the computation
        save: bool
            save the data
        start: int
            Starting position in the database. All values before start will be
            ignored.
        stop: int
            Stopping position in the database. All values after stop will be
            ignored.
        number_of_configurations: int
            The number of uniformly sampled configuration between start and
            stop to be used for the RDF.
        kwargs:
            overide_n_batches: int
                    override the automatic batch size calculation
            use_tf_function : bool
                    If true, tf.function is used in the calculation.
        """
        # set args that will affect the computation result
        self.args = Args(
            start=start,
            stop=stop,
            data_range=1,
            correlation_time=1,
            max_bond_length=max_bond_length,
            number_of_configurations=number_of_configurations,
            molecules=molecules,
            species=species,
            atom_selection=atom_selection,
        )
        # args parsing that will not affect the computation result
        # usually performance or plotting
        self.plot = plot

        # kwargs parsing
        self.use_tf_function = kwargs.pop("use_tf_function", False)
        self.override_n_batches = kwargs.get("batches")
        self.tqdm_limit = kwargs.pop("tqdm", 10)

    def check_input(self):
        """
        Check the input of the call method and store defaults if needed.

        Returns
        -------
        Updates class attributes if required.
        """
        if self.args.stop is None:
            self.args.stop = self.experiment.number_of_configurations - 1

        if self.args.number_of_configurations == -1:
            self.args.number_of_configurations = (
                self.experiment.number_of_configurations - 1
            )

        # Get the correct species out.
        if self.args.species is None:
            if self.args.molecules:
                self.args.species = list(self.experiment.molecules)
            else:
                self.args.species = list(self.experiment.species)

        self.sample_configurations = np.linspace(
            self.args.start,
            self.args.stop,
            self.args.number_of_configurations,
            dtype=np.int,
        )  # choose sampled configurations

    def _correct_batch_properties(self):
        """
        We must fix the batch size parameters set by the parent class.

        Returns
        -------
        Updates the parent class.
        """
        if self.batch_size > self.args.number_of_configurations:
            self.batch_size = self.args.number_of_configurations
            self.n_batches = 1
        else:
            self.n_batches = int(self.args.number_of_configurations / self.batch_size)

        if self.override_n_batches is not None:
            self.n_batches = self.override_n_batches

    def plot_data(self, data):
        """Plot the atoms vs neighbors"""
        # TODO: it creates the plot in a "sub-plot", is this created in the calculator class?

        colors = Category10[
            10
        ]  # create a color iterator with 10 colors, we do not need more.
        data_neighbors = data["System"]
        fig = figure(x_axis_label=self.x_label, y_axis_label=self.y_label)
        for idx, (neighbors, time_evolution) in enumerate(data_neighbors.items()):
            time_steps = np.arange(len(time_evolution))
            # Add result and hover tool
            fig.line(
                time_steps,
                np.array(time_evolution),
                color=colors[int(idx)],  # we use the neighbor number to set the color.
                # legend labels are the number of neighbors.
                legend_label=f"{neighbors}",
            )
            fig.add_tools(HoverTool())
            self.plot_array.append(fig)

    def _post_operation_processes(self, lst_dict_neighbors):
        """
        call the post-op processes
        Returns
        -------

        """
        dict_neighbors_result = {
            k: [] for k in range(0, 9)
        }  # we cannot have more than 8 bonds for an atom...

        for dict_neighbors in lst_dict_neighbors:
            for n_neighbors, count in dict_neighbors.items():
                dict_neighbors_result[n_neighbors].append(count)

        # Clean-up dictionary removing entries with only zeros.
        dict_neighbors_result = {
            k: v for k, v in dict_neighbors_result.items() if sum(v) != 0
        }

        # I believe so far it is simpler to just add everything
        logging.debug(dict_neighbors_result)

        # TODO: revise if this should be system, or instead we create another section in the DB.
        #  But I do not know how to do it.
        self.queue_data(data=dict_neighbors_result, subjects=["System"])

    def prepare_computation(self):
        """
        Run the steps necessary to prepare for the RDF computation.

        Returns
        -------
        dict_keys : list
                dict keys to use when selecting data from the output.
        split_arr : np.ndarray
                Array of indices to load from the database split into sub-arrays which
                fulfill the necessary batch size.
        batch_tqdm : bool
                If true, the main tqdm loop over batches is disabled and only the
                mini-batch loop will be displayed.
        """

        path_list = [
            join_path(item, self.loaded_property.name) for item in self.args.species
        ]
        self._prepare_managers(path_list)

        # batch loop correction
        self._correct_batch_properties()

        # Get the correct dict keys.
        dict_keys = []
        for item in self.args.species:
            dict_keys.append(str.encode(join_path(item, self.loaded_property.name)))

        # Split the configurations into batches.
        split_arr = np.array_split(self.sample_configurations, self.n_batches)

        # Turn off the tqdm for certain scenarios.
        batch_tqdm = self.tqdm_limit > self.n_batches

        return dict_keys, split_arr, batch_tqdm

    def _format_data(self, batch: tf.Tensor, keys: list) -> tf.Tensor:
        """
        Format the loaded data for use this calculators.

        It requires a matrix with the positions. The generator will load a default
        dict oriented type. This method restructures the data to be used in the
        calculator.

        Parameters
        ----------
        batch : tf.Tensor
                A batch of data to transform.
        keys : list
                Dict keys to extract from the data.

        Returns
        -------
        data : tf.Tensor
                data tensor of the shape (n_atoms * n_species, n_configurations, 3)

        """
        formatted_data = []
        for item in keys:
            formatted_data.append(batch[item])

        if len(self.args.species) == 1:
            return tf.cast(formatted_data[0], dtype=self.dtype)
        else:
            return tf.cast(tf.concat(formatted_data, axis=0), dtype=self.dtype)

    def _compute_neighbors(self, adj_dict: dict) -> dict:
        """
        This method computes the number of neighbors from the adjacency matrix.

        Parameters
        ----------
        adj_dict: adjacency dictionary computed from self.create_adj_dict

        Returns
        -------
        dict: dictionary with the counts of each number of neighbors

        """
        dict_neighbors = {
            k: 0 for k in range(0, 9)
        }  # we cannot have more than 8 bonds for an atom...

        for _, value in adj_dict.items():
            len_list = len(value)
            dict_neighbors[len_list] += 1

        logging.debug(dict(dict_neighbors))

        return dict(dict_neighbors)  # recast the defaultdict back into a normal dict.

    def create_adj_dict(
        self,
        positions: tf.Tensor,
        r: float,
        leaf_size: int = 10,
        box_size: float | list = None,
    ) -> dict:
        """
        Method to create adjacency dictionary from xyz atomic positions,
        It uses the KDTree algorithm
        Parameters
        ----------
        positions: tf.Tensor
                timestep of positions
        box_size: float or list
                Size of the box
        leaf_size: int
                Number of leafs in the KDTree
        r: float
                Minimum distance to consider an atom as a neighbour
        Returns
        -------
        Dict: Dictionary of atomic adjacencies
        """
        tree = KDTree(
            positions, leafsize=leaf_size, boxsize=box_size
        )  # Create KDTree, avoid searching all the space and splits the domain.
        all_nn_indices = tree.query_ball_point(
            positions, r, workers=5
        )  # Calculates neighbours within radius r of a point.
        adj_dict = {}
        for count, item in enumerate(all_nn_indices):
            adj_dict[count] = item  # Populate adjacency dictionary

        for node, nodes in adj_dict.items():
            if node in nodes:
                nodes.remove(node)  # Remove duplicates
        adj_dict = {k: set(v) for k, v in adj_dict.items()}

        logging.info("Adjacency matrix created.")
        return adj_dict

    def run_calculator(self):
        """
        Run the analysis.

        Returns
        -------

        """

        self.check_input()

        dict_keys, split_arr, batch_tqm = self.prepare_computation()

        # Get the batch dataset
        batch_ds = self.get_batch_dataset(
            subject_list=self.args.species, loop_array=split_arr, correct=True
        )

        lst_dict_neighbors = []
        # Loop over the batches.
        for idx, batch in tqdm(enumerate(batch_ds), ncols=70, total=self.n_batches):
            positions_tensor = self._format_data(batch=batch, keys=dict_keys)
            shape = tf.shape(positions_tensor)
            n_configs = shape[1]
            for config in range(n_configs):
                adj_dict = self.create_adj_dict(
                    positions_tensor[:, config, :], self.args.max_bond_length
                )
                dict_neighbors = self._compute_neighbors(adj_dict)
                lst_dict_neighbors.append(dict_neighbors)

        self._post_operation_processes(lst_dict_neighbors)
