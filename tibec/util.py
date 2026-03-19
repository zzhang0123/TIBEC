"""
Utility functions for beam pattern processing.

Provides helper functions for stereographic projection, beam intensity
rescaling, spherical interpolation, angular windowing, and FFT-based
baseline beam extraction.
"""

import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

from scipy.fftpack import dctn, fftn, fftshift, fftfreq
from scipy.interpolate import RectBivariateSpline, RectSphereBivariateSpline


def plane2sphere_v2(x, y):
    """
    Map 2D planar coordinates to spherical coordinates via stereographic
    projection.

    Uses the standard stereographic projection formulae:

        phi   = arctan2(y, x),  corrected to [0, 2*pi)
        theta = 2 * arctan(sqrt(x^2 + y^2) / 2)

    Parameters
    ----------
    x : ndarray
        Planar x-coordinates.
    y : ndarray
        Planar y-coordinates (same shape as x).

    Returns
    -------
    phi : ndarray
        Azimuthal angle in [0, 2*pi) radians.
    theta : ndarray
        Polar angle in radians.
    """
    phi = np.arctan2(y, x)
    for i in range(len(phi)):
        if  phi[i] < 0.:
            phi[i] += 2. * np.pi
    theta = 2. * np.arctan(np.sqrt(x ** 2 + y ** 2) / 2.)
    return phi, theta


def Beam_scaled(beam, theta_coord):
    """
    Apply intensity rescaling to correct for the stereographic projection
    Jacobian.

    The rescaling factor accounts for the area distortion introduced by
    stereographic projection:

        factor = cos(theta/2)^5 / (cos(theta/2) - sin(theta/2))

    Parameters
    ----------
    beam : ndarray, shape (N_x, N_y, N_components)
        Beam data on the stereographic plane. Last axis is the number of
        components (e.g., Stokes parameters or polarisation channels).
    theta_coord : ndarray, shape (N_x, N_y)
        Polar angle theta at each grid point (radians).

    Returns
    -------
    ndarray, shape (N_x, N_y, N_components)
        Rescaled beam data.
    """
    aux_theta = theta_coord/2.
    intensity_rescaling_factor = np.cos(aux_theta) ** 5 / (np.cos(aux_theta) - np.sin(aux_theta))
    B_matrix = intensity_rescaling_factor[:, :, np.newaxis] * beam
    return B_matrix


def interpolation(Ndim, phi, theta, beam, target_phi, target_theta):
    """
    Interpolate beam data from a regular spherical grid to arbitrary target
    positions using ``RectSphereBivariateSpline``.

    The interpolation is performed component-by-component along the last axis
    of ``beam``.

    Parameters
    ----------
    Ndim : int
        Output grid dimension. The result is reshaped to (Ndim, Ndim, ...).
    phi : ndarray, shape (N_phi,)
        Azimuthal angle grid values in radians. The first value (phi=0) is
        excluded to avoid duplication at the boundary.
    theta : ndarray, shape (N_theta,)
        Polar angle grid values in radians (colatitude).
    beam : ndarray, shape (N_theta, N_phi, N_components)
        Beam data on the regular (theta, phi) grid.
    target_phi : ndarray
        Target azimuthal angles (radians), flattened to length Ndim*Ndim.
    target_theta : ndarray
        Target polar angles (radians), flattened to length Ndim*Ndim.

    Returns
    -------
    ndarray, shape (Ndim, Ndim, N_components)
        Interpolated beam data at the target positions.
    """
    Beams_interpolated = np.zeros(shape=(Ndim, Ndim, beam.shape[-1]))
    for i in np.arange(beam.shape[-1]):
        interp = RectSphereBivariateSpline(theta[:], phi[1:], beam[:, 1:, i],
                                           pole_values=(beam[0, 0, i],None),
                                           pole_exact=(True,False)
                                          )
        Beams_interpolated[:, :, i] = interp.ev(target_theta, target_phi).reshape(Ndim, Ndim)
    return Beams_interpolated


def directional_window(Beam, theta_coords, theta_max=75., alpha=0.05):
    """
    Apply a smooth angular windowing function to a beam pattern.

    Pixels with theta < theta_max are multiplied by a smooth exponential
    taper. Pixels with theta >= theta_max are set to zero. The taper is:

        w(theta) = exp(-alpha * (1/(theta - theta_max)^2 - 1/theta_max^2))

    This creates a smooth transition to zero at theta_max.

    .. warning::
        This function modifies ``Beam`` in place.

    Parameters
    ----------
    Beam : ndarray, shape (N_x, N_y, N_components)
        Beam data to window. Modified in place.
    theta_coords : ndarray, shape (N_x, N_y)
        Polar angle at each grid point (in the same units as theta_max,
        typically degrees).
    theta_max : float, optional
        Cutoff angle beyond which the beam is set to zero. Default: 75.
    alpha : float, optional
        Taper steepness parameter. Default: 0.05.

    Returns
    -------
    ndarray, shape (N_x, N_y, N_components)
        Windowed beam data (same array as input, modified in place).
    """
    for i in range(theta_coords.shape[0]):
        for j in range(theta_coords.shape[1]):
            if theta_coords[i,j] < theta_max:
                Beam[i, j, :] *= np.exp(-alpha*(1/(theta_coords[i,j]-theta_max)**2 - 1/theta_max**2))
            else:
                Beam[i, j, :] = 0.
    return Beam


def q_matrix(x_fft_coords, y_fft_coords, baseline_length):
    """
    Compute the sub-grid indices and coordinate meshgrid for extracting a
    baseline-centred region from FFT coordinates.

    Determines a square sub-region of the FFT grid centred on the origin,
    with half-width determined by the baseline length.

    Parameters
    ----------
    x_fft_coords : ndarray, shape (N,)
        FFT x-axis coordinate values (e.g., from ``fftfreq``).
    y_fft_coords : ndarray, shape (N,)
        FFT y-axis coordinate values.
    baseline_length : float
        Baseline length in the same units as the FFT coordinates.

    Returns
    -------
    half_dim : int
        Half-width of the extracted sub-grid (in grid points).
    meshgrid : tuple of ndarray
        (X, Y) meshgrid of the sub-region FFT coordinates.
    """
    xk_res = x_fft_coords[1]-x_fft_coords[0]
    half_dim = np.int64((len(x_fft_coords) -1)/2. - baseline_length/xk_res)
    ind_x = np.absolute(x_fft_coords).argmin()
    ind_x_l = ind_x-half_dim
    ind_x_r = ind_x+half_dim+1
    ind_y = np.absolute(y_fft_coords).argmin()
    ind_y_l = ind_y-half_dim
    ind_y_r = ind_y+half_dim+1
    return  half_dim, np.meshgrid(x_fft_coords[ind_x_l:ind_x_r], y_fft_coords[ind_x_l:ind_x_r])


def FFT_Beam_matrix(FFT_beam, x_fft_coords, y_fft_coords, half_dim, baseline_length, angle):
    """
    Extract a sub-matrix of the FFT beam at a given baseline length and angle.

    Shifts the extraction centre to the position corresponding to the baseline
    vector (baseline_length, angle) in the FFT plane:

        q_x = baseline_length * cos(angle)
        q_y = baseline_length * sin(angle)

    Parameters
    ----------
    FFT_beam : ndarray, shape (N_y, N_x, N_components)
        2D FFT of the beam pattern.
    x_fft_coords : ndarray, shape (N_x,)
        FFT x-axis coordinate values.
    y_fft_coords : ndarray, shape (N_y,)
        FFT y-axis coordinate values.
    half_dim : int
        Half-width of the sub-grid to extract (from ``q_matrix``).
    baseline_length : float
        Baseline length in FFT coordinate units.
    angle : float
        Baseline orientation angle in **degrees**.

    Returns
    -------
    ndarray, shape (2*half_dim+1, 2*half_dim+1, N_components)
        Extracted sub-matrix of the FFT beam centred at the baseline position.
    """
    q_x_shift = baseline_length * np.cos(np.deg2rad(angle))
    q_y_shift = baseline_length * np.sin(np.deg2rad(angle))
    ind_x = np.absolute(x_fft_coords - q_x_shift).argmin()
    ind_y = np.absolute(y_fft_coords - q_y_shift).argmin()
    return FFT_beam[ind_y - half_dim: ind_y + half_dim+1, ind_x - half_dim: ind_x + half_dim + 1, :]
