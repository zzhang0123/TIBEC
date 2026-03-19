"""
Coordinate system transformations for TIBEC.

This module implements the transformations between the five coordinate systems
used in beam pattern analysis:

1. Antenna spherical (phi_a, theta_a)
2. Antenna Cartesian (x_a, y_a, z_a)
3. Horizontal (local) Cartesian (x_h, y_h, z_h) — South, East, Zenith
4. Equatorial Cartesian (x_e, y_e, z_e)
5. Equatorial spherical (phi_e, theta_e) — where theta_e = pi/2 - DEC, phi_e = RA

All angles are in radians. LST is in seconds.
Spherical coordinate arrays use (phi, theta) ordering throughout.

See ``TIBEC documentation.pdf`` for the full mathematical derivations.
"""

import numpy as np


def coordinate_mapping_sph2car(spherical_coords_array):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Implements the standard mapping (Section 2 of the documentation):

        x = sin(theta) * cos(phi)
        y = sin(theta) * sin(phi)
        z = cos(theta)

    with unit radius (r = 1).

    Parameters
    ----------
    spherical_coords_array : ndarray, shape (..., 2)
        Spherical coordinates with the last axis being (phi, theta) in radians.
        Supports arbitrary leading dimensions (e.g., (N_sky, 2) or
        (N_LST, N_sky, 2)).

    Returns
    -------
    ndarray, shape (..., 3)
        Cartesian coordinates (x, y, z). Leading dimensions are preserved.
    """
    sph_coords_array = spherical_coords_array.reshape(-1, 2)
    nsky = sph_coords_array.shape[0]
    theta_array = sph_coords_array[:, 1]
    phi_array = sph_coords_array[:, 0]
    car_coords_array = np.zeros(shape=(nsky, 3))
    car_coords_array[:, 0] = np.sin(theta_array) * np.cos(phi_array)
    car_coords_array[:, 1] = np.sin(theta_array) * np.sin(phi_array)
    car_coords_array[:, 2] = np.cos(theta_array)
    return car_coords_array.reshape(spherical_coords_array.shape[:-1]+(3,))


def coordinate_mapping_car2sph(car_coords_array):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Inverse of ``coordinate_mapping_sph2car``. Computes:

        theta = arctan2(sqrt(x^2 + y^2), z)
        phi   = arctan2(y, x),  corrected to [0, 2*pi)

    Parameters
    ----------
    car_coords_array : ndarray, shape (..., 3)
        Cartesian coordinates (x, y, z). Supports arbitrary leading dimensions.

    Returns
    -------
    ndarray, shape (..., 2)
        Spherical coordinates (phi, theta) in radians, with phi in [0, 2*pi).
        Leading dimensions are preserved.
    """
    XYZ_array = car_coords_array.reshape(-1, 3)
    nsky = XYZ_array.shape[0]
    sph_coords_array = np.zeros(shape=(nsky, 2))
    aux = np.sqrt(XYZ_array[:, 0] ** 2 + XYZ_array[:, 1] ** 2)
    sph_coords_array[:, 1] = np.arctan2(aux, XYZ_array[:, 2])
    phi_array = np.arctan2(XYZ_array[:, 1], XYZ_array[:, 0])

    def correct_phi(phi):
        if phi < 0:
            result = phi + 2 * np.pi
        else:
            result = phi
        return result

    func = np.vectorize(correct_phi)
    sph_coords_array[:, 0] = func(phi_array)
    return sph_coords_array.reshape(car_coords_array.shape[:-1]+(2,))


def rotation_matrix_local2eq(LSTs, latitude):
    """
    Build the rotation matrix from horizontal (local) to equatorial Cartesian
    coordinates, R_{h->eq}.

    This is the matrix R_{eq-h}(theta, phi) from Eq. 4 of the documentation,
    evaluated at theta = pi/2 - latitude and phi = LST * (pi/12) / 3600.

    Since R_{eq-h} is orthogonal, R_{h->eq} = R_{eq-h}^T, so this function
    returns the transpose of Eq. 4.

    Parameters
    ----------
    LSTs : array_like
        Local Sidereal Time(s) in **seconds**. Scalar or 1D array.
        Internally converted to radians via: phi = (LST / 3600) * (pi / 12).
    latitude : float
        Antenna latitude in radians.

    Returns
    -------
    ndarray, shape (N_LST, 3, 3)
        Rotation matrices, one per LST value.
    """
    LSTs = np.array(LSTs).flatten()
    phi_array = (LSTs / 3600.) * (np.pi / 12.0)
    the_array = (np.pi / 2 - latitude) * np.ones_like(phi_array)

    sin_phi, cos_phi = np.sin(phi_array), np.cos(phi_array)
    sin_the, cos_the = np.sin(the_array), np.cos(the_array)
    matrix = np.zeros(shape=(LSTs.shape[0], 3, 3))
    matrix[:, 0, 0] = cos_the * cos_phi
    matrix[:, 1, 0] = cos_the * sin_phi
    matrix[:, 2, 0] = -sin_the
    matrix[:, 0, 1] = -sin_phi
    matrix[:, 1, 1] = cos_phi
    matrix[:, 0, 2] = sin_the * cos_phi
    matrix[:, 1, 2] = sin_the * sin_phi
    matrix[:, 2, 2] = cos_the
    return matrix


def rotation_matrix_eq2local(LSTs, latitude):
    """
    Build the rotation matrix from equatorial to horizontal (local) Cartesian
    coordinates, R_{eq->h}.

    Since the rotation is orthogonal: R_{eq->h} = R_{h->eq}^T.

    Parameters
    ----------
    LSTs : array_like
        Local Sidereal Time(s) in **seconds**.
    latitude : float
        Antenna latitude in radians.

    Returns
    -------
    ndarray, shape (N_LST, 3, 3)
        Rotation matrices, one per LST value.
    """
    matrix = rotation_matrix_local2eq(LSTs, latitude)
    return np.swapaxes(matrix, -1, -2)


def rotation_matrix_sph2car(sph_coords_array):
    """
    Build the rotation matrix R_{s->c} that transforms vector components from
    the spherical basis (theta-hat, phi-hat) to the Cartesian basis (x, y, z).

    This returns only the last two columns of the full R_{s->c} matrix (Eq. 6),
    since the radial component (E_r) is zero for far-field radiation:

        R_{s->c}[:, 1:] = [[cos(theta)*cos(phi), -sin(phi)],
                            [cos(theta)*sin(phi),  cos(phi)],
                            [-sin(theta),          0       ]]

    Parameters
    ----------
    sph_coords_array : ndarray, shape (..., 2)
        Spherical coordinates (phi, theta) in radians.

    Returns
    -------
    ndarray, shape (..., 3, 2)
        Transformation matrices from (theta, phi) components to (x, y, z).
    """
    def rot_mat(phi_the_array):
        sin_phi, cos_phi = np.sin(phi_the_array[0]), np.cos(phi_the_array[0])
        sin_the, cos_the = np.sin(phi_the_array[1]), np.cos(phi_the_array[1])
        matr = np.zeros(shape=(3, 3))
        matr[:, 0] = sin_the * cos_phi, sin_the * sin_phi, cos_the
        matr[:, 1] = cos_the * cos_phi, cos_the * sin_phi, -sin_the
        matr[:, 2] = -sin_phi, cos_phi, 0.0
        return matr[:, 1:]
    result = np.apply_along_axis(rot_mat, -1, sph_coords_array)
    return result


def rotation_matrix_car2sph(sph_coords_array):
    """
    Build the rotation matrix R_{c->s} that transforms vector components from
    the Cartesian basis (x, y, z) to the spherical basis (r, theta, phi).

    This is the full transpose of R_{s->c} (Eq. 6), returning a (3, 3) matrix
    per sky point. Ref: Eq. 11 uses [R_{s->c}]^{-1} = [R_{s->c}]^T.

    Parameters
    ----------
    sph_coords_array : ndarray, shape (..., 2)
        Spherical coordinates (phi, theta) in radians.

    Returns
    -------
    ndarray, shape (..., 3, 3)
        Transformation matrices from (x, y, z) to (r, theta, phi).
    """
    def rot_mat(phi_the_array):
        sin_phi, cos_phi = np.sin(phi_the_array[0]), np.cos(phi_the_array[0])
        sin_the, cos_the = np.sin(phi_the_array[1]), np.cos(phi_the_array[1])
        matr = np.zeros(shape=(3, 3))
        matr[:, 0] = sin_the * cos_phi, sin_the * sin_phi, cos_the
        matr[:, 1] = cos_the * cos_phi, cos_the * sin_phi, -sin_the
        matr[:, 2] = -sin_phi, cos_phi, 0.0
        return matr.T
    result = np.apply_along_axis(rot_mat, -1, sph_coords_array)
    return result


def pointing_matrix(x_Axis, z_Axis):
    """
    Build the pointing matrix P that maps antenna Cartesian coordinates to
    horizontal (local) Cartesian coordinates (Eq. 3, 7, 8).

    The pointing matrix is constructed from the antenna's x-axis and z-axis
    directions expressed in horizontal coordinates. The y-axis is derived by
    the right-hand rule: y = z x x. All axes are normalised to unit vectors.

    The columns of P are the unit antenna axes [x_hat, y_hat, z_hat], so that:

        [x_h, y_h, z_h]^T = P @ [x_a, y_a, z_a]^T     (Eq. 7)

    For example, x_Axis = [1, 0, 0] means the antenna x-axis points to true
    South, and z_Axis = [0, 0, 1] means the antenna z-axis points to Zenith.

    Parameters
    ----------
    x_Axis : array_like, shape (3,)
        Antenna x-axis direction in horizontal coordinates [S, E, Z].
    z_Axis : array_like, shape (3,)
        Antenna z-axis direction in horizontal coordinates [S, E, Z].

    Returns
    -------
    ndarray, shape (3, 3)
        Orthogonal pointing matrix P. Its transpose P^T maps from horizontal
        to antenna coordinates.
    """
    xAxis = np.array(x_Axis)
    zAxis = np.array(z_Axis)
    yAxis = np.cross(zAxis, xAxis)
    unit_xAxis = xAxis / np.linalg.norm(xAxis)
    unit_zAxis = zAxis / np.linalg.norm(zAxis)
    unit_yAxis = yAxis / np.linalg.norm(yAxis)
    point_Mat = np.zeros(shape=(3, 3))
    point_Mat[:, 0] = unit_xAxis
    point_Mat[:, 1] = unit_yAxis
    point_Mat[:, 2] = unit_zAxis
    return point_Mat


def R_ant_h(beam_angles, yy):
    """
    Build antenna-to-horizontal rotation matrices for a set of beam tilt angles.

    This is a special case of the pointing matrix for an antenna tilted by a
    given angle in the x-z plane (if yy=False) or y-z plane (if yy=True).
    Used by ``EFieldBeamAngle`` for computing beams at different pointing angles.

    For yy=False (x-z plane tilt by angle b):
        x_axis = [cos(b), 0, sin(b)],  z_axis = [-sin(b), 0, cos(b)]

    For yy=True (y-z plane tilt by angle b):
        x_axis = [0, cos(b), sin(b)],  z_axis = [0, -sin(b), cos(b)]

    Parameters
    ----------
    beam_angles : array_like
        Beam tilt angles in **radians**.
    yy : bool
        If False, tilt in the x-z plane (default for x-polarisation feed).
        If True, tilt in the y-z plane (for y-polarisation / orthogonal feed).

    Returns
    -------
    Rot_ant2h : ndarray, shape (N_angles, 3, 3)
        Rotation matrices from antenna to horizontal coordinates (P^T).
    Rot_h2ant : ndarray, shape (N_angles, 3, 3)
        Rotation matrices from horizontal to antenna coordinates (P).
    """
    beam_angles = np.array(beam_angles)
    Rot_ant2h = np.zeros(shape=(len(beam_angles), 3, 3))
    Rot_h2ant = np.zeros(shape=(len(beam_angles), 3, 3))
    for i in range(len(beam_angles)):
        b_ang = beam_angles[i]
        cos_b_ang, sin_b_ang  = np.cos(b_ang), np.sin(b_ang)
        if yy:
            alignment_x = np.array([0., cos_b_ang, sin_b_ang])
            alignment_z = np.array([0., -sin_b_ang, cos_b_ang])
        else:
            alignment_x = np.array([cos_b_ang,  0., sin_b_ang])
            alignment_z = np.array([-sin_b_ang, 0., cos_b_ang])
        pointing_mat = pointing_matrix(alignment_x, alignment_z)
        Rot_ant2h[i] = pointing_mat.T
        Rot_h2ant[i] = pointing_mat
    return Rot_ant2h, Rot_h2ant
