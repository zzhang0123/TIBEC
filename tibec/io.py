"""
Data loading and convenience functions for TIBEC.

Provides utilities for reading E-field data files and a high-level function
for computing Full Stokes beams directly from a data file.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# Pauli matrices for Stokes computation (same as in farfieldtransfer)
_pauli_array = 0.5 * np.array([
    [[1., 0.], [0., 1.]],     # I
    [[1., 0.], [0., -1.]],    # Q
    [[0., 1.], [1., 0.]],     # U
    [[0., -1.j], [1.j, 0.]],  # V
])


def load_efield_txt(path, n_phi, n_theta, n_freq=1, degrees=True):
    """
    Load E-field data from a far-field text file (FEKO/CST format).

    Reads the standard 6-column format:

        phi  theta  Re(E_theta)  Im(E_theta)  Re(E_phi)  Im(E_phi)

    Lines starting with ``//``, ``#``, or matching the grid size header
    (e.g., ``361 181``) are treated as comments and skipped.

    Parameters
    ----------
    path : str
        Path to the E-field text file.
    n_phi : int
        Number of phi samples per frequency block.
    n_theta : int
        Number of theta samples per frequency block.
    n_freq : int, optional
        Number of frequency blocks in the file. Default: 1.
    degrees : bool, optional
        If True (default), convert phi/theta coordinates from degrees to
        radians.

    Returns
    -------
    coords : ndarray, shape (n_phi * n_theta, 2)
        Antenna spherical coordinates (phi, theta). In radians if
        ``degrees=True``, otherwise in the file's native units.
    e_field : ndarray, shape (n_freq, n_phi, n_theta, 2) or (n_phi, n_theta, 2)
        Complex E-field data. Last axis is (E_theta, E_phi).
        If ``n_freq == 1``, the frequency axis is squeezed out.

    Examples
    --------
    Load single-frequency HIRAX data:

    >>> coords, efield = load_efield_txt("data/HIRAX_201-Copy1.txt", 361, 181)
    >>> efield.shape
    (361, 181, 2)

    Load multi-frequency REACH data (26 frequencies):

    >>> coords, efield = load_efield_txt("data/REACH_Efield.txt", 73, 37, n_freq=26)
    >>> efield.shape
    (26, 73, 37, 2)
    """
    n_sky = n_phi * n_theta
    comment_header = f"{n_phi} {n_theta}"
    comments = ('//', '#', comment_header)

    # Load coordinates from the first frequency block
    coords = np.loadtxt(path, comments=comments, usecols=(0, 1),
                        max_rows=n_sky).reshape(-1, 2)
    if degrees:
        coords = np.deg2rad(coords)

    # Load E-field (real + imaginary parts)
    e_field_real = np.loadtxt(path, comments=comments, usecols=(2, 4))
    e_field_imag = np.loadtxt(path, comments=comments, usecols=(3, 5))
    e_field = (e_field_real + 1j * e_field_imag).reshape(n_freq, n_phi, n_theta, 2)

    if n_freq == 1:
        e_field = e_field[0]  # squeeze frequency axis

    return coords, e_field


def compute_stokes_beams(efield_path, n_phi, n_theta, n_freq=1, freq_ind=0,
                         nside=64):
    """
    Compute Full Stokes (I, Q, U, V) beam maps from an E-field data file.

    This is a high-level convenience function that performs the complete
    pipeline:

    1. Load the E-field from a text file.
    2. Interpolate it onto a HEALPix sky grid using ``RegularGridInterpolator``.
    3. Compute the Stokes beam matrix: B_p(s) = Re[E*(s) . sigma_p . E(s)].

    This function assumes the antenna is at the north pole with standard
    alignment (x-axis = South, z-axis = Zenith), so that the antenna frame
    and equatorial frame coincide. No coordinate rotation is applied.

    Parameters
    ----------
    efield_path : str
        Path to the E-field text file (6-column FEKO/CST format).
    n_phi : int
        Number of phi samples per frequency block.
    n_theta : int
        Number of theta samples per frequency block.
    n_freq : int, optional
        Number of frequency blocks. Default: 1.
    freq_ind : int, optional
        Frequency channel index to use (only applies when n_freq > 1).
        Default: 0.
    nside : int, optional
        HEALPix Nside parameter for the output sky grid. Default: 64
        (giving 49152 pixels).

    Returns
    -------
    stokes_beams : ndarray, shape (N_pix, 4)
        Real-valued Stokes beam maps. Columns: [I, Q, U, V].
        N_pix = 12 * nside^2.
    sky_coords : ndarray, shape (N_pix, 2)
        HEALPix sky coordinates (phi, theta) in radians.

    Examples
    --------
    >>> stokes, coords = compute_stokes_beams("data/HIRAX_201-Copy1.txt", 361, 181)
    >>> stokes.shape
    (49152, 4)

    Visualise Stokes I with healpy:

    >>> import healpy as hp
    >>> hp.mollview(stokes[:, 0], title="Stokes I")
    """
    import healpy as hp

    # Load E-field data
    coords, e_field = load_efield_txt(efield_path, n_phi, n_theta,
                                      n_freq=n_freq)

    # Select frequency channel if multi-frequency
    if n_freq > 1:
        e_field_2d = e_field[freq_ind]  # shape: (n_phi, n_theta, 2)
    else:
        e_field_2d = e_field  # shape: (n_phi, n_theta, 2)

    # Extract the regular grid axes
    phi_array = np.unique(coords[:, 0])
    theta_array = np.unique(coords[:, 1])

    # Build HEALPix sky coordinates
    npix = hp.nside2npix(nside)
    sky_coords = np.zeros((npix, 2))
    sky_coords[:, 1], sky_coords[:, 0] = hp.pix2ang(nside, np.arange(npix))

    # Interpolate E-field onto HEALPix grid
    e_field_sky = np.zeros((npix, 2), dtype=complex)
    for i in range(2):
        interp = RegularGridInterpolator(
            [phi_array, theta_array],
            e_field_2d[:, :, i],
            bounds_error=False,
            fill_value=0.
        )
        e_field_sky[:, i] = interp(sky_coords)

    # Compute Stokes beams: B_p = Re[E* . sigma_p . E]
    stokes_beams = np.einsum("sl, plm, sm -> sp",
                             np.conjugate(e_field_sky),
                             _pauli_array,
                             e_field_sky).real

    return stokes_beams, sky_coords
