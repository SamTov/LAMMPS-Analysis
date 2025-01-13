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
"""

import logging

import numpy as np
import tensorflow as tf
from tqdm import tqdm

log = logging.getLogger(__name__)


def get_triu_indicies(n_atoms):
    """
    Version of np.triu_indices with k=1.

    Returns
    -------
    Returns a vector of size (2, None) instead of a tuple of two values like
    np.triu_indices

    """
    bool_mat = tf.ones((n_atoms, n_atoms), dtype=tf.bool)
    # Just construct a boolean true matrix the size of one time_step
    indices = tf.where(~tf.linalg.band_part(bool_mat, -1, 0))
    # Get the indices of the lower triangle (without diagonals)
    indices = tf.cast(indices, dtype=tf.int32)  # Get the correct dtype
    return tf.transpose(indices)  # Return the transpose for convenience later


def get_neighbour_list(positions: tf.Tensor, cell=None, batch_size=None) -> tf.Tensor:
    """
    Generate the neighbour list.

    Parameters
    ----------
    positions: tf.Tensor
        Tensor with shape (n_confs, n_atoms, 3) representing the coordinates
    cell: list
        If periodic boundary conditions are used, please supply the cell dimensions,
        e.g. [13.97, 13.97, 13.97]. If the cell is provided minimum image convention
        will be applied!
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

    def get_rij_mat(positions, triu_mask, cell):
        """
        Use the upper triangle of the virtual r_ij matrix constructed of
        n_atoms * n_atoms matrix and subtract the transpose to get all distances once!
        If PBC are used, apply the minimum image convention.
        """
        r_ij_mat = tf.gather(positions, triu_mask[0], axis=1) - tf.gather(
            positions, triu_mask[1], axis=1
        )
        if cell:
            r_ij_mat -= tf.math.rint(r_ij_mat / cell) * cell
        return r_ij_mat

    n_atoms = positions.shape[1]
    triu_mask = get_triu_indicies(n_atoms)

    if batch_size is not None:
        if not positions.shape[0] % batch_size == 0:
            msg = (
                "positions must be evenly divisible by batch_size, but are"
                f" {positions.shape[0]} and {batch_size}"
            )
            log.error(msg)
            raise RuntimeError(msg)

        for positions_batch in tf.split(positions, batch_size):
            yield get_rij_mat(positions_batch, triu_mask, cell)
    else:
        yield get_rij_mat(positions, triu_mask, cell)


# @tf.function
def get_triplets(
    full_r_ij: tf.Tensor, r_cut: float, n_atoms: int, n_batches=200, disable_tqdm=True
) -> tf.Tensor:
    """Compute the triple indices within a cutoff.

    Mostly vectorized method to compute the triples inside the cutoff sphere. Therefore
    a matrix of all possible distances *r_ijk* over all timesteps
    (n_timesteps, n_atoms, n_atoms, n_atoms) is constructed. The first layer is
    identical to the r_ij matrix, all consecutive layers are shifted along the "j" axis
    which leads to the full r_ijk matrix.

    To check for a triple, the depth can be converted into the third particle, yielding
    an index over all *ijk* indices

    Parameters
    ----------
    full_r_ij: tf.Tensor
        The full distance matrix (r_ij and r_ji) with the shape
        (n_timesteps, n_atoms, n_atoms)
    r_cut: float
        The cutoff for the maximal triple distance
    n_atoms: int
        Number of atoms
    n_batches: int
        Number of batches to split the computation of the triples in.
        A low number of batches can result in memory issues.

    Returns
    -------
    tf.Tensor

    Warnings
    ---------
    Using the @tf.function decorator can speed things up! But it can also lead to
    memory issues.

    """
    if n_batches >= n_atoms:
        n_batches = n_atoms - 1
    r_ij = tf.norm(full_r_ij, axis=-1)
    r_ij = tf.cast(r_ij, dtype=tf.float16)  # Using float16 for maximal memory safety.
    r_ij = tf.where(r_ij == 0, tf.ones_like(r_ij) * r_cut, r_ij)

    batches = np.array_split(np.arange(1, n_atoms), n_batches)
    triples = []
    # batches would be [(1, 100), (101, 200), (201, 300), ...]
    for batch in tqdm(batches, ncols=70, disable=disable_tqdm):
        r_ijk = tf.TensorArray(tf.float16, size=len(batch))
        for idx, atom in enumerate(batch):
            tmp = tf.roll(r_ij, shift=-atom, axis=2)
            r_ijk = r_ijk.write(idx, tmp)
        r_ijk = r_ijk.stack()

        r_ijk = tf.transpose(r_ijk, perm=(1, 0, 2, 3))  # shape is (t, n, i, j)
        intermediate_indices = tf.where(
            tf.math.logical_and(r_ij[:, None] < r_cut, r_ijk < r_cut)
        )

        n_atoms = tf.cast(n_atoms, dtype=tf.int64)

        t, n, i, j = tf.unstack(intermediate_indices, axis=1)
        k = j + n + batch[0]  # this comes from tf.roll.
        k = tf.where(k >= n_atoms, k - n_atoms, k)
        triples.append(tf.stack([t, i, j, k], axis=1))

    return tf.concat(triples, axis=0)
