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


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / tf.expand_dims(tf.norm(vector, axis=-1), -1)


def angle_between(v1, v2, acos=True):
    """Returns the angle in radians between vectors 'v1' and 'v2'."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    # return tf.math.acos(tf.clip_by_value(tf.einsum("ijk, ijk -> ij", v1_u, v2_u),
    # -1.0, 1.0))
    if acos:
        return tf.math.acos(
            tf.clip_by_value(tf.einsum("ij, ij -> i", v1_u, v2_u), -1.0, 1.0)
        )
    else:
        return tf.einsum("ij, ij -> i", v1_u, v2_u)


def get_angles(r_ij_mat, indices, acos=True):
    """
    Compute the cosine angle for the given triples.

    Using :math theta = acos(r_ij * r_ik / (|r_ij| * |r_ik|))

    Parameters
    ----------
    acos: bool
        Apply tf.math.acos to the output if true. default true.
    r_ij_mat: tf.Tensor
        r_ij matrix. Shape is (n_timesteps, n_atoms, n_atoms, 3)
    indices: tf.Tensor
        Indices of the triples. Shape is (triples, 4) where a single element is composed
        of (timestep, idx, jdx, kdx)

    Returns
    -------
    tf.Tensor: Tensor with the shape (triples)

    """
    r_ij = tf.gather_nd(
        r_ij_mat, tf.stack([indices[:, 0], indices[:, 1], indices[:, 2]], axis=1)
    )
    r_ik = tf.gather_nd(
        r_ij_mat, tf.stack([indices[:, 0], indices[:, 1], indices[:, 3]], axis=1)
    )

    return (
        angle_between(r_ij, r_ik, acos),
        tf.linalg.norm(r_ij, axis=-1) * tf.linalg.norm(r_ik, axis=-1),
    )


@tf.function(experimental_relax_shapes=True)
def apply_minimum_image(r_ij, box_array):
    """

    Parameters
    ----------
    r_ij: tf.Tensor
        an r_ij matrix of size (batches, atoms, 3)
    box_array: tf.Tensor
        a box array (3,)

    Returns
    -------

    """
    return r_ij - tf.math.rint(r_ij / box_array) * box_array


def get_partial_triu_indices(n_atoms: int, m_atoms: int, idx: int) -> tf.Tensor:
    """Calculate the indices of a slice of the triu values.

    Parameters
    ----------
    n_atoms: total number of atoms in the system
    m_atoms: size of the slice (horizontal)
    idx: start index of slize

    Returns
    -------
    tf.Tensor

    """
    bool_mat = tf.ones((m_atoms, n_atoms), dtype=tf.bool)
    bool_vector = ~tf.linalg.band_part(bool_mat, -1, idx)  # rename!

    indices = tf.where(bool_vector)
    indices = tf.cast(indices, dtype=tf.int32)  # is this large enough?!
    indices = tf.transpose(indices)
    return indices


def apply_system_cutoff(tensor: tf.Tensor, cutoff: float) -> tf.Tensor:
    """
    Enforce a cutoff on a tensor.

    Parameters
    ----------
    tensor : tf.Tensor
    cutoff : flaot

    """
    cutoff_mask = tf.cast(tf.less(tensor, cutoff), dtype=tf.bool)  # Construct the mask

    return tf.boolean_mask(tensor, cutoff_mask)


def cartesian_to_spherical_coordinates(
    point_cartesian, name="cartesian_to_spherical_coordinates"
):
    """
    References:
    ----------
    https://www.tensorflow.org/graphics/api_docs/python/tfg/math/math_helpers/cartesian_to_spherical_coordinates.

    Function to transform Cartesian coordinates to spherical coordinates.
    This function assumes a right handed coordinate system with `z` pointing up.
    When `x` and `y` are both `0`, the function outputs `0` for `phi`. Note that
    the function is not smooth when `x = y = 0`.

    Note:
    ----
      In the following, A1 to An are optional batch dimensions.

    Args:
    ----
      point_cartesian: A tensor of shape `[A1, ..., An, 3]`. In the last
        dimension, the data follows the `x`, `y`, `z` order.
      eps: A small `float`, to be added to the denominator. If left as `None`, its
        value is automatically selected using `point_cartesian.dtype`.
      name: A name for this op. Defaults to "cartesian_to_spherical_coordinates".

    Returns:
    -------
      A tensor of shape `[A1, ..., An, 3]`. The last dimensions contains
      (`r`,`theta`,`phi`), where `r` is the sphere radius, `theta` is the polar
      angle and `phi` is the azimuthal angle. Returns `NaN` gradient if x = y = 0.

    """
    with tf.name_scope(name):
        point_cartesian = tf.convert_to_tensor(value=point_cartesian)

        x, y, z = tf.unstack(point_cartesian, axis=-1)
        radius = tf.norm(tensor=point_cartesian, axis=-1)
        theta = tf.acos(tf.clip_by_value(tf.math.divide_no_nan(z, radius), -1.0, 1.0))
        phi = tf.atan2(y, x)
        return tf.stack((radius, theta, phi), axis=-1)


def spherical_to_cartesian_coordinates(
    point_spherical, name="spherical_to_cartesian_coordinates"
):
    """
    References:
    ----------
    https://www.tensorflow.org/graphics/api_docs/python/tfg/math/math_helpers/spherical_to_cartesian_coordinates.

    Function to transform Cartesian coordinates to spherical coordinates.

    Note:
    ----
      In the following, A1 to An are optional batch dimensions.

    Args:
    ----
      point_spherical: A tensor of shape `[A1, ..., An, 3]`. The last dimension
        contains r, theta, and phi that respectively correspond to the radius,
        polar angle and azimuthal angle; r must be non-negative.
      name: A name for this op. Defaults to "spherical_to_cartesian_coordinates".

    Raises:
    ------
      tf.errors.InvalidArgumentError: If r, theta or phi contains out of range
      data.

    Returns:
    -------
      A tensor of shape `[A1, ..., An, 3]`, where the last dimension contains the
      cartesian coordinates in x,y,z order.

    """
    with tf.name_scope(name):
        point_spherical = tf.convert_to_tensor(value=point_spherical)

        r, theta, phi = tf.unstack(point_spherical, axis=-1)
        tmp = r * tf.sin(theta)
        x = tmp * tf.cos(phi)
        y = tmp * tf.sin(phi)
        z = r * tf.cos(theta)
        return tf.stack((x, y, z), axis=-1)


def get2dHistogram(x, y, value_range, nbins=100, dtype=tf.dtypes.int32):
    """
    Bins x, y coordinates of points onto simple square 2d histogram.

    Given the tensor x and y:
    x: x coordinates of points
    y: y coordinates of points
    this operation returns a rank 2 `Tensor`
    representing the indices of a histogram into which each element
    of `values` would be binned. The bins are equal width and
    determined by the arguments `value_range` and `nbins`.


    Parameters
    ----------
    x:  Numeric `Tensor`.
    y: Numeric `Tensor`.
    value_range[0] lims for x
    value_range[1] lims for y

    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
    dtype:  dtype for returned histogram.

    References
    ----------
    https://gist.github.com/isentropic/a86effab2c007e86912a50f995cac52b

    """
    x_range = value_range[0]
    y_range = value_range[1]

    histy_bins = tf.histogram_fixed_width_bins(y, y_range, nbins=nbins, dtype=dtype)

    H = tf.map_fn(
        lambda i: tf.histogram_fixed_width(x[histy_bins == i], x_range, nbins=nbins),
        tf.range(nbins),
    )
    return H  # Matrix!
