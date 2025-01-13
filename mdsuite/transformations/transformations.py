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
Parent class for the transformations.
"""

from __future__ import annotations

import abc
import collections.abc
import copy
import logging
import os
import time
import typing
from typing import TYPE_CHECKING, Union

import numpy as np
import tensorflow as tf
import tqdm

import mdsuite.database.simulation_database
from mdsuite.database.data_manager import DataManager
from mdsuite.database.simulation_database import Database
from mdsuite.memory_management.memory_manager import MemoryManager
from mdsuite.utils import DatasetKeys
from mdsuite.utils.meta_functions import join_path

if TYPE_CHECKING:
    from mdsuite.experiment import Experiment


class CannotFindTransformationError(Exception):
    """Exception for when a transformation cannot be found."""

    pass


class CannotFindPropertyError(Exception):
    """Exception for when a property cannot be found."""

    pass


class Transformations:
    """
    Parent class for MDSuite transformations.

    Attributes
    ----------
    _database : Database
            database class object for data loading and storing
    _experiment : object
            Experiment class instance to update
    batch_size : int
            batch size for the computation
    n_batches : int
            Number of batches to be looped over
    remainder : int
            Remainder amount to add after the batches are looped over.
    data_manager : DataManager
            data manager for handling the data transfer
    memory_manager : MemoryManager
            memory manager for the computation.

    """

    def __init__(
        self,
        input_properties: typing.Iterable[
            mdsuite.database.simulation_database.PropertyInfo
        ] = None,
        output_property: mdsuite.database.simulation_database.PropertyInfo = None,
        scale_function=None,
        dtype=tf.float64,
    ):
        """
        Init of the transformer base class.

        Parameters
        ----------
        input_properties : typing.Iterable[
                    mdsuite.database.simulation_database.PropertyInfo]
            The properties needed to perform the transformation.
            e.g. unwrapped positions for the wrap_coordinates transformation.
            Properties from this list are provided to the self.transform_batch(),
            in which the actual transformation happens.
        output_property : mdsuite.database.simulation_database.PropertyInfo
            The property that is the result of the transformation
        scale_function :
            specifies memory requirements of the transformation
        dtype :
            data type of the processed values

        """
        self._experiment = None
        self._database = None

        self.input_properties = input_properties
        self.output_property = output_property
        self.logger = logging.getLogger(__name__)
        self.scale_function = scale_function
        self.dtype = dtype

        self.batch_size: int
        self.n_batches: int
        self.remainder: int

        self.offset = 0

        self.data_manager: DataManager
        self.memory_manager: MemoryManager

    @property
    def database(self):
        """Update the database.

        replace for https://github.com/zincware/MDSuite/issues/404
        """
        if self._database is None:
            self._database = Database(self.experiment.database_path / "database.hdf5")
        return self._database

    @property
    def experiment(self) -> Experiment:
        """TODO replace for https://github.com/zincware/MDSuite/issues/404."""
        return self._experiment

    @experiment.setter
    def experiment(self, value):
        self._experiment = value

    def _run_dataset_check(self, path: str):
        """
        Check to see if the database dataset already exists. If it does, the
        transformation should extend the dataset and add data to the end of
        it rather than try to add data.

        Parameters
        ----------
        path : str
                dataset path to check.

        Returns
        -------
        outcome : bool
                If True, the dataset already exists and should be extended.
                If False, a new dataset should be built.

        """
        return self.database.check_existence(path)

    def _save_output(
        self,
        data: Union[tf.Tensor, np.array],
        index: int,
        data_structure: dict,
    ):
        """
        Save the tensor_values into the database_path
        # todo for the future: this should not be part of the transformation.
        # the transformation should yield a batch and the experiment should take care of
        # storing it in the correct place, just as with file inputs

        Returns
        -------
        saves the tensor_values to the database_path.

        """
        # turn data into trajectory chunk
        # data_structure is dict {'/path/to/property':{'indices':irrelevant,
        #                           'columns':deduce->deduce n_dims, 'length':n_particles}
        species_list = []
        # data structure only has 1 element
        key, val = list(data_structure.items())[0]
        path = str(copy.copy(key))
        path.rstrip("/")
        path = path.split("/")
        prop_name = path[-1]
        sp_name = path[-2]
        n_particles = val.get("length")
        if n_particles is None:
            try:
                # if length is not available try indices next
                n_particles = len(val.get("indices"))
            except TypeError:
                raise TypeError("Could not determine number of particles")
        if len(np.shape(data)) == 2:
            # data not for multiple particles, instead one value for all
            # -> create the n_particle axis
            data = data[np.newaxis, :, :]
        prop = mdsuite.database.simulation_database.PropertyInfo(
            name=prop_name, n_dims=len(val["columns"])
        )
        species_list.append(
            mdsuite.database.simulation_database.SpeciesInfo(
                name=sp_name, properties=[prop], n_particles=n_particles
            )
        )
        chunk = mdsuite.database.simulation_database.TrajectoryChunkData(
            chunk_size=np.shape(data)[1], species_list=species_list
        )
        # data comes from transformation with time in 1st axis, add_data needs it
        # in 0th axis
        chunk.add_data(
            data=np.swapaxes(data, 0, 1),
            config_idx=0,
            species_name=sp_name,
            property_name=prop_name,
        )

        try:
            self.database.add_data(chunk=chunk)
        except OSError:
            """
            This is used because in Windows and in WSL we got the error that
            the file was still open while it should already be closed. So, we
            wait, and we add again.
            """
            time.sleep(0.5)
            self.database.add_data(chunk=chunk)

    def _prepare_monitors(self, data_path: Union[list, np.array]):
        """
        Prepare the tensor_values and memory managers.

        Parameters
        ----------
        data_path : list
                List of tensor_values paths to load from the hdf5
                database_path.

        Returns
        -------

        """
        self.memory_manager = MemoryManager(
            data_path=data_path,
            database=self.database,
            memory_fraction=0.5,
            scale_function=self.scale_function,
            offset=self.offset,
        )
        (
            self.batch_size,
            self.n_batches,
            self.remainder,
        ) = self.memory_manager.get_batch_size()
        self.data_manager = DataManager(
            data_path=data_path,
            data_range=1,
            database=self.database,
            batch_size=self.batch_size,
            n_batches=self.n_batches,
            remainder=self.remainder,
            offset=self.offset,
        )

    def _prepare_database_entry(self, species: str, system_tensor=False):
        """
        Add or extend the dataset in which the transformation result is stored.

        Parameters
        ----------
        species : str
                Species for which transformation is performed

        Returns
        -------
        tensor_values structure for use in saving the tensor_values to the
        database_path.

        """
        if system_tensor:
            output_length = 1
            path = join_path(DatasetKeys.OBSERVABLES, self.output_property.name)
        else:
            try:
                output_length = self.experiment.species[species].n_particles
            except KeyError:
                output_length = self.experiment.molecules[species].n_particles
            path = join_path(species, self.output_property.name)
        n_dims = self.output_property.n_dims

        existing = self._run_dataset_check(path)
        if existing:
            old_shape = self.database.get_data_size(path)
            resize_structure = {
                path: (
                    output_length,
                    self.experiment.number_of_configurations - old_shape[0],
                    n_dims,
                )
            }
            self.offset = old_shape[0]
            self.database.resize_datasets(resize_structure)

        else:
            number_of_configurations = self.experiment.number_of_configurations
            dataset_structure = {path: (output_length, number_of_configurations, n_dims)}
            self.database.add_dataset(dataset_structure)

        data_structure = {
            path: {
                "indices": np.s_[:],
                "columns": list(range(n_dims)),
                "length": output_length,
            }
        }

        return data_structure

    def find_property_per_config(self, sp_name, prop) -> typing.Union[None, str]:
        would_be_path = join_path(sp_name, prop.name)
        if self.database.check_existence(would_be_path):
            return would_be_path
        else:
            return None

    def find_property_single_val(self, sp_name, prop):
        # TODO: properties in species_dict are all lowercase,
        # whereas the properties in the database are upper case
        try:
            species_info = self.experiment.species[sp_name]
        except KeyError:
            species_info = self.experiment.molecules[sp_name]
        per_sp_value = getattr(species_info, prop.name.lower(), None)
        if per_sp_value is not None:
            return per_sp_value
        # if not species specific, try to find it system-wide
        # todo need a dictionary of these values instead of the experiment property
        try:
            return self.experiment.__getattribute__(prop.name.lower())
        except AttributeError:
            return None

    def get_prop_through_transformation(self, sp_name, prop):
        # todo prevent infinite recursion
        # (e.g. unwrap_pos calls wrap_pos calls unwrap_pos calls ...)
        from mdsuite.transformations.transformation_dict import (
            property_to_transformation_dict,
        )

        if prop in property_to_transformation_dict.keys():
            trafo_cls = property_to_transformation_dict[prop]
            if not isinstance(trafo_cls, typing.Iterable):
                # simple case: only one trafo for the property
                trafo = trafo_cls()
                self.experiment.cls_transformation_run(trafo, species=[sp_name])
            else:
                # go through the list until one works
                one_trafo_worked = False
                for trafo_class in trafo_cls:
                    trafo = trafo_class()
                    try:
                        self.experiment.cls_transformation_run(trafo, species=[sp_name])
                    except CannotFindPropertyError:
                        pass
                    else:
                        one_trafo_worked = True
                        break
                if not one_trafo_worked:
                    raise CannotFindTransformationError(
                        f"was asked to get '{prop.name}' for '{sp_name}'. "
                        f"There are transformations to get this property ({trafo_cls}), "
                        "but none of them have the required data"
                    )
            return self.find_property_per_config(sp_name, prop)
        else:
            raise CannotFindTransformationError(
                f"was asked to get '{prop.name}' for '{sp_name}', but there is no"
                " transformation to get that property"
            )

    def get_generator_type_spec_and_const_data(self, species_names):
        type_spec = {}
        const_input_data = {}

        for species_name in species_names:
            const_input_data[species_name] = {}
            for prop in self.input_properties:
                # find out if the requested input data is there for all time steps
                # for the requested species
                path = self.find_property_per_config(species_name, prop)
                if path is not None:
                    type_spec[str.encode(path)] = tf.TensorSpec(
                        shape=(None, None, prop.n_dims), dtype=self.dtype
                    )
                # if not, fall back to const value for that species
                else:
                    val = self.find_property_single_val(species_name, prop)
                    if val is not None:
                        # give single value the same dimensionality as if it was there
                        # for each time step and particle (i.e. add 2 axes)
                        if not isinstance(val, collections.abc.Iterable):
                            val = [val]
                        val = tf.convert_to_tensor(val, dtype=self.dtype)
                        val = val[None, None, :]
                        const_input_data[species_name].update({prop.name: val})
                    # if not there, try to produce the data
                    else:
                        try:
                            path = self.get_prop_through_transformation(
                                species_name, prop
                            )
                            type_spec[str.encode(path)] = tf.TensorSpec(
                                shape=(None, None, prop.n_dims), dtype=self.dtype
                            )
                        except CannotFindTransformationError:
                            raise CannotFindPropertyError(
                                "While performing transformation"
                                f" '{self.output_property.name}': Property '{prop.name}'"
                                f" for species '{species_name}' cannot be found in the"
                                " simulation database nor in the simulation metadata, nor"
                                " can it be obtained by a transformation"
                            )

        return type_spec, const_input_data

    @abc.abstractmethod
    def run_transformation(self, species: typing.Iterable[str] = None):
        raise NotImplementedError


class SingleSpeciesTrafo(Transformations):
    """
    Base class for transformations where the transformation is applied to each species
    separately.
    """

    def run_transformation(self, species: typing.Iterable[str] = None):
        """
        Perform the batching and data loading for the transformation,
        then calls transform_batch

        Parameters
        ----------
        species : Iterable[str]
            Names of the species on which to perform the transformation.

        Returns
        -------

        """
        # species should be provided by caller (the experiment), for now we use the usual
        # pseudoglobal variable
        if species is None:
            species = self.experiment.species.keys()

        for species_name in species:
            # this check should be done by the caller
            if self.database.check_existence(
                os.path.join(species_name, self.output_property.name)
            ):
                self.logger.info(
                    f"{self.output_property.name} already exists for {species_name}, "
                    "skipping transformation"
                )
                continue

            output_data_structure = self._prepare_database_entry(species_name)
            type_spec, const_input_data = self.get_generator_type_spec_and_const_data(
                [species_name]
            )
            const_input_data = const_input_data[species_name]

            self._prepare_monitors(list(type_spec.keys()))
            batch_generator, batch_generator_args = self.data_manager.batch_generator()
            type_spec.update(
                {
                    str.encode("data_size"): tf.TensorSpec(shape=(), dtype=tf.int32),
                }
            )
            data_set = tf.data.Dataset.from_generator(
                batch_generator, args=batch_generator_args, output_signature=type_spec
            )
            data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)
            carryover = None
            for index, batch_dict in tqdm.tqdm(
                enumerate(data_set),
                ncols=70,
                desc=(
                    f"Applying transformation '{self.output_property.name}' to"
                    f" '{species_name}'"
                ),
                total=self.n_batches,
            ):
                # remove species information from batch:
                # the transformation only has to know about the property
                # ideally, the keys of the batch dict are already PropertyInfo instances
                batch_dict.pop(str.encode("data_size"))
                batch_dict_wo_species = {}
                for key, val in batch_dict.items():
                    batch_dict_wo_species[key.decode().split("/")[-1]] = val
                batch_dict_wo_species.update(const_input_data)
                ret = self.transform_batch(batch_dict_wo_species, carryover=carryover)
                if isinstance(ret, tuple):
                    transformed_batch, carryover = ret
                else:
                    transformed_batch = ret
                self._save_output(
                    data=transformed_batch,
                    data_structure=output_data_structure,
                    index=index * self.batch_size,
                )

    @abc.abstractmethod
    def transform_batch(
        self, batch: typing.Dict[str, tf.Tensor], carryover: typing.Any = None
    ) -> typing.Union[tf.Tensor, typing.Tuple[tf.Tensor, typing.Any]]:
        """
        Do the actual transformation.

        Parameters
        ----------
        batch : dict
            The batch to be transformed. structure is
            {'Property1': tansordata, ...}
        carryover : any
            if the transformation batching is only possible with carryover,
            this argument will provide it.

        Returns
        -------
        Either the transformed batch (tf.Tensor)
        Or tuple of (<transformed batch>, <carryover>),
        where the carryover can have any type.
        The carryover will be used as the optional argument for the next batch

        """
        raise NotImplementedError("transformation of a batch must be implemented")


class MultiSpeciesTrafo(Transformations):
    """
    Base class for all transformations, where information of multiple species is combined
    in the transformation of a new property.
    """

    def run_transformation(self, species: typing.Iterable[str] = None) -> None:
        """
        Perform the batching and data loading for the transformation,
        then calls transform_batch

        Parameters
        ----------
        species : Iterable[str]
            Names of the species on which to perform the transformation.

        Returns
        -------

        """
        # species should be provided by caller (the experiment), for now we use the usual
        # pseudoglobal variable
        if species is None:
            species = self.experiment.species.keys()

        # this check should be done by the caller
        if self.database.check_existence(
            os.path.join(self.output_property.name, self.output_property.name)
        ):
            self.logger.info(
                f"{self.output_property.name} already exists for this experiment, "
                "skipping transformation"
            )
            return

        output_data_structure = self._prepare_database_entry(
            self.output_property.name, system_tensor=True
        )
        type_spec, const_input_data = self.get_generator_type_spec_and_const_data(species)
        self._prepare_monitors(list(type_spec.keys()))
        batch_generator, batch_generator_args = self.data_manager.batch_generator()
        type_spec.update(
            {
                str.encode("data_size"): tf.TensorSpec(shape=(), dtype=tf.int32),
            }
        )
        data_set = tf.data.Dataset.from_generator(
            batch_generator, args=batch_generator_args, output_signature=type_spec
        )
        data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)
        carryover = None
        for index, batch_dict in tqdm.tqdm(
            enumerate(data_set),
            ncols=70,
            desc=f"Applying transformation '{self.output_property.name}'",
            total=self.n_batches,
        ):
            batch_dict.pop(str.encode("data_size"))
            batch_dict_hierachical = {sp_name: {} for sp_name in species}
            for key, val in batch_dict.items():
                sp_name, prop_name = key.decode().split("/")
                batch_dict_hierachical[sp_name][prop_name] = val
            for sp_name in batch_dict_hierachical.keys():
                batch_dict_hierachical[sp_name].update(const_input_data[sp_name])
            ret = self.transform_batch(batch_dict_hierachical, carryover=carryover)
            if isinstance(ret, tuple):
                transformed_batch, carryover = ret
            else:
                transformed_batch = ret
            self._save_output(
                data=transformed_batch,
                data_structure=output_data_structure,
                index=index * self.batch_size,
            )

    @abc.abstractmethod
    def transform_batch(
        self,
        batch: typing.Dict[str, typing.Dict[str, tf.Tensor]],
        carryover: typing.Any = None,
    ) -> tf.Tensor | typing.Tuple[tf.Tensor, typing.Any]:
        """
        Do the actual transformation.

        Parameters
        ----------
        batch : dict
            The batch to be transformed. structure is
            {'Species1': {'Property1': tensordata, ...}, ...}
        carryover : any
            if the transformation batching is only possible with carryover,
            this argument will provide it, see below.

        Returns
        -------
        Either the transformed batch (tf.Tensor)
        Or tuple of (<transformed batch>, <carryover>),
        where the carryover can have any type.
        The carryover will be used as the optional argument for the next batch

        """
        raise NotImplementedError("transformation of a batch must be implemented")
