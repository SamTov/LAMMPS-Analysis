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

import tensorflow as tf

from .helpers import triu_indices as compute_triu


class NLLayer(tf.keras.layers.Layer):
    """Convert positions to an r_ij distance matrix."""

    def __init__(self, dense: bool = True, **kwargs):
        """
        Parameters
        ----------
        dense: bool
            Return the flat_rij or a dense r_ij

        """
        super().__init__(**kwargs)
        self.dense = dense

    def call(self, inputs, *args, **kwargs):
        """Convert positions to r_ij distance matrix.

        Parameters
        ----------
        inputs: dict
            containing the keys positions and cell with e.g., positions shape
            (None, n_atoms, 3)

        Returns
        -------
        flat_rij, triu_indices, n_atoms: tf.Tensor, tf.Tensor, tf.Tenosr
            The distances flattend out, of shape (x, 3) and the corresponding indices
            in the r_ij matrix of shape (x, 2) as well as the number of atoms

        """
        positions = tf.cast(inputs["positions"], self.dtype)
        cell = tf.cast(inputs["cell"], self.dtype)

        n_atoms = tf.shape(positions)[1]
        triu_indices = compute_triu(n_atoms, k=1)

        flat_rij = tf.gather(positions, triu_indices[:, 0], axis=1) - tf.gather(
            positions, triu_indices[:, 1], axis=1
        )

        cell = tf.linalg.diag_part(cell)

        flat_rij -= tf.math.rint(flat_rij / cell[:, None]) * cell[:, None]

        if self.dense:

            def to_dense(flat_rij):
                """Convert the flattened output to a dense r_ij matrix.

                Parameters
                ----------
                flat_rij: tf.Tensor
                    The flat r_ij matrix

                Returns
                -------
                r_ij: tf.Tensor
                    A dense r_ij tensor

                """
                r_ij = tf.scatter_nd(
                    indices=triu_indices, updates=flat_rij, shape=(n_atoms, n_atoms, 3)
                )
                r_ij -= tf.transpose(r_ij, (1, 0, 2))

                return r_ij

            return tf.vectorized_map(to_dense, flat_rij)
        return flat_rij, triu_indices, n_atoms
