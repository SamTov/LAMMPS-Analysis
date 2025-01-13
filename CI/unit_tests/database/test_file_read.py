"""Test MDSuite file reading."""

import numpy as np

import mdsuite
import mdsuite.file_io.script_input as script_input
from mdsuite.database.simulation_database import (
    PropertyInfo,
    SpeciesInfo,
    TrajectoryChunkData,
    TrajectoryMetadata,
)

err_decimal = 5


def get_species_list(n_species=1, prop_names=["Positions"], n_particles=17, n_dims=3):
    props = []
    for prop_name in prop_names:
        props.append(PropertyInfo(name=prop_name, n_dims=n_dims))
    ret = []
    for i in range(n_species):
        ret.append(
            SpeciesInfo(
                name=f"sp_test_{i}", mass=1.1, n_particles=n_particles, properties=props
            )
        )

    return ret


def test_species_info_equal():
    properties_0 = [
        PropertyInfo(name="test0", n_dims=2),
        PropertyInfo(name="test1", n_dims=2),
    ]
    properties_1 = [
        PropertyInfo(name="test0", n_dims=2),
        PropertyInfo(name="test1", n_dims=987654),
    ]

    assert properties_0[0] == properties_1[0]

    sp_info_0 = SpeciesInfo(name="test0", mass=3, n_particles=17, properties=properties_0)
    sp_info_1 = SpeciesInfo(name="test0", mass=3, n_particles=17, properties=properties_0)
    sp_info_2 = SpeciesInfo(name="test0", mass=3, n_particles=17, properties=properties_1)

    assert sp_info_0 == sp_info_1
    assert sp_info_0 != sp_info_2


def test_traj_chunk_data():
    prop_name = "my_property"
    sp_list = get_species_list(
        n_species=2, prop_names=[prop_name], n_particles=5, n_dims=2
    )
    n_configs = 11
    chunk_size = 83
    sp_name = sp_list[1].name
    data = np.random.rand(*(n_configs, 5, 2))

    chunk = TrajectoryChunkData(species_list=sp_list, chunk_size=chunk_size)

    # write data to the end of the chunk
    chunk.add_data(
        data=data,
        species_name=sp_name,
        property_name=prop_name,
        config_idx=chunk_size - n_configs,
    )

    chunk_data = chunk.get_data()
    my_prop_data = chunk_data[sp_name][prop_name]
    assert my_prop_data.shape == (chunk_size, 5, 2)

    np.testing.assert_array_almost_equal(
        my_prop_data[: chunk_size - n_configs, :, :],
        np.zeros((chunk_size - n_configs, 5, 2)),
    )
    np.testing.assert_array_almost_equal(
        my_prop_data[chunk_size - n_configs :, :, :], data
    )


def test_read_script_input(tmp_path):
    n_configs = 10
    n_parts = 4
    n_dims = 2
    time_step = 0.1
    sp_name = "test_species"
    positions = np.random.rand(*(n_configs, n_parts, n_dims))
    velocities = np.random.rand(*(n_configs, n_parts, n_dims))

    properties = [
        PropertyInfo(name="Positions", n_dims=n_dims),
        PropertyInfo(name="Velocities", n_dims=n_dims),
    ]

    species_list = [
        SpeciesInfo(name=sp_name, n_particles=n_parts, mass=1234, properties=properties)
    ]

    metadata = TrajectoryMetadata(
        species_list=species_list,
        n_configurations=n_configs,
        sample_rate=1,
        box_l=3 * [1.1],
    )
    data = TrajectoryChunkData(species_list=species_list, chunk_size=n_configs)
    data.add_data(positions, 0, sp_name, "Positions")
    data.add_data(velocities, 0, sp_name, "Velocities")

    proc = script_input.ScriptInput(data=data, metadata=metadata, name="test_name")

    project = mdsuite.Project(name="test_proj", storage_path=tmp_path)
    project.add_experiment(name="test_experiment", timestep=time_step)
    exp = project.experiments["test_experiment"]
    exp.add_data(proc)

    pos_loaded = np.swapaxes(
        exp.load_matrix(species=[sp_name], property_name="Positions")[
            "test_species/Positions"
        ].numpy(),
        0,
        1,
    )
    vel_loaded = np.swapaxes(
        exp.load_matrix(species=[sp_name], property_name="Velocities")[
            "test_species/Velocities"
        ].numpy(),
        0,
        1,
    )

    np.testing.assert_array_almost_equal(positions, pos_loaded, decimal=err_decimal)
    np.testing.assert_array_almost_equal(velocities, vel_loaded, decimal=err_decimal)
