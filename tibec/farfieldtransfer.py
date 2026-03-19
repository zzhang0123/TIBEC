"""
Far-field beam pattern computation in equatorial coordinates.

This module provides classes for transforming antenna E-field patterns from
antenna-local coordinates into equatorial (RA/DEC) coordinates and computing
Full Stokes (I, Q, U, V) beam matrices.

Two classes are provided:

- ``E_field``: LST-based beam computation for instruments with explicit antenna
  alignment (e.g., REACH). Supports multiple frequencies and time-dependent
  beam rotation via Local Sidereal Time.

- ``EFieldBeamAngle``: Beam-angle-based computation for instruments where the
  beam is parameterised by tilt angles (e.g., HIRAX). Uses HEALPix
  nearest-neighbour interpolation for the E-field lookup.

The Stokes beam matrix is computed via the Pauli matrices:

    B_p(s) = E*(s) . sigma_p . E(s)

where p = {I, Q, U, V} and sigma_p are the 2x2 Pauli matrices.

See ``TIBEC documentation.pdf`` for full mathematical derivations.
"""

from tibec.transformation import *
from scipy.interpolate import RegularGridInterpolator


import healpy as hp


# --- Pauli matrices for Stokes parameter computation ---
# Used to compute Full Stokes beams: B_p = E* sigma_p E
# Index 0 = I, 1 = Q, 2 = U, 3 = V
Pauli_I = 0.5 * np.array([[1., 0.],
                          [0., 1.]])
Pauli_Q = 0.5 * np.array([[1., 0.],
                          [0., -1.]])
Pauli_U = 0.5 * np.array([[0., 1.],
                          [1., 0.]])
Pauli_V = 0.5 * np.array([[0., -1.j],
                          [1.j, 0.]])

pauli_array = np.array([Pauli_I, Pauli_Q, Pauli_U, Pauli_V])


class E_field():
    """
    LST-based E-field beam model in equatorial coordinates.

    This class handles the full transformation of an antenna's complex E-field
    pattern from antenna-local coordinates to equatorial coordinates at given
    Local Sidereal Times (LSTs). It supports multi-frequency E-field data and
    explicit antenna alignment specification.

    The transformation chain is:

        equatorial spherical → equatorial Cartesian
        → (R_{eq->h}) → horizontal Cartesian
        → (P^T = R_{h->ant}) → antenna Cartesian
        → antenna spherical (for E-field lookup)

    The field components are transformed via the corresponding basis rotation
    matrices along the same chain.

    Parameters
    ----------
    ant_coords : ndarray, shape (N_sky, 2)
        Antenna spherical coordinates (phi, theta) in radians, where
        N_sky = N_phi * N_theta. Used to extract the regular phi/theta grids.
    e_field_data : ndarray, shape (N_freq, N_phi, N_theta, 2)
        Complex E-field data. The last axis holds the two polarisation
        components (E_theta, E_phi).
    alignment_x : array_like, shape (3,)
        Antenna x-axis direction in horizontal coordinates [South, East, Zenith].
        Example: [1, 0, 0] means x-axis points to true South.
    alignment_z : array_like, shape (3,)
        Antenna z-axis direction in horizontal coordinates [South, East, Zenith].
        Example: [0, 0, 1] means z-axis points to Zenith.
    ant_lat : float
        Antenna latitude in radians.

    Attributes
    ----------
    field : ndarray
        The stored E-field data.
    nfreq : int
        Number of frequency channels.
    phi_array : ndarray
        Unique phi values from the antenna coordinate grid (radians).
    theta_array : ndarray
        Unique theta values from the antenna coordinate grid (radians).
    pointing_mat : ndarray, shape (3, 3)
        The pointing matrix P (see ``pointing_matrix``).
    R_ant2h : ndarray, shape (3, 3)
        Rotation from antenna to horizontal coordinates (P^T).
    R_h2ant : ndarray, shape (3, 3)
        Rotation from horizontal to antenna coordinates (P).
    antenna_latitude : float
        Antenna latitude in radians.
    """

    def __init__(self, ant_coords, e_field_data, alignment_x, alignment_z, ant_lat):
        self.pointing_mat = pointing_matrix(alignment_x, alignment_z)
        self.R_ant2h = self.pointing_mat.T
        self.R_h2ant = self.pointing_mat
        self.nfreq = e_field_data.shape[0]
        self.field = e_field_data
        self.antenna_latitude = ant_lat
        self.phi_array = np.unique(ant_coords[:, 0])
        self.theta_array = np.unique(ant_coords[:, 1])
        print("The far-field object has been initialized!")

    def map_sky_coords_to_ant_coords(self, LSTs, eq_sph_coords):
        """
        Map equatorial sky coordinates to antenna coordinates at given LSTs.

        Applies the full coordinate and field-component transformation chain:

        1. Equatorial spherical → equatorial Cartesian
        2. R_{eq->h}(LST, lat) → horizontal Cartesian
        3. P^T (R_{h->ant}) → antenna Cartesian
        4. Antenna Cartesian → antenna spherical

        Also computes the field transformation matrix that rotates E-field
        vector components from the antenna spherical basis to the equatorial
        spherical basis.

        Einsum index conventions: t = LST, s = sky pixel, i/j/k/l = matrix.

        Parameters
        ----------
        LSTs : array_like
            Local Sidereal Times in seconds.
        eq_sph_coords : ndarray, shape (N_sky, 2)
            Equatorial spherical coordinates (phi, theta) in radians.

        Returns
        -------
        ant_sph_coords : ndarray, shape (N_LST, N_sky, 2)
            Sky positions mapped to antenna spherical coordinates.
        field_trans_mat : ndarray, shape (N_LST, N_sky, 3, 2)
            Field transformation operator. Transforms E-field components from
            antenna spherical (theta, phi) basis to equatorial spherical
            (r, theta, phi) basis.
        """
        eq_car_coords = coordinate_mapping_sph2car(eq_sph_coords)  # shape: (N_sky, 3)
        R_eq2h = rotation_matrix_eq2local(LSTs, self.antenna_latitude)  # shape: (N_LST, 3, 3)
        ant_car_coords = np.einsum("ij,tjk,sk -> tsi", self.R_h2ant, R_eq2h, eq_car_coords)
        ant_sph_coords = coordinate_mapping_car2sph(ant_car_coords)  # shape: (N_LST, N_sky, 2)

        R_sc = rotation_matrix_sph2car(ant_sph_coords)  # shape: (N_LST, N_sky, 3, 2)
        R_h2eq = rotation_matrix_local2eq(LSTs, self.antenna_latitude) # shape: (N_LST, 3, 3)
        R_cs_eq = rotation_matrix_car2sph(eq_sph_coords)  # shape: (N_sky, 3, 3)
        field_trans_mat = np.einsum("sij, tjk, tskl -> tsil",
                                    R_cs_eq, R_h2eq@self.R_ant2h, R_sc) # shape: (N_LST, N_sky, 3, 2)
        return ant_sph_coords, field_trans_mat

    def make_interp(self, points, freq_ind):
        """
        Interpolate the E-field at arbitrary antenna coordinates using linear
        interpolation on the regular (phi, theta) grid.

        Uses ``scipy.interpolate.RegularGridInterpolator`` with
        ``bounds_error=False`` and ``fill_value=0`` (points outside the antenna
        grid are set to zero).

        Parameters
        ----------
        points : ndarray, shape (N_LST, N_sky, 2)
            Antenna spherical coordinates (phi, theta) at which to evaluate
            the E-field.
        freq_ind : int
            Frequency channel index into ``self.field``.

        Returns
        -------
        ndarray, shape (N_LST, N_sky, 2), dtype complex
            Interpolated E-field components (E_theta, E_phi).
        """
        result = np.zeros(shape=points.shape,  dtype=complex)
        for i in np.arange(2):
            interp = RegularGridInterpolator([self.phi_array, self.theta_array],
                                             self.field[freq_ind, :, :, i],
                                             bounds_error=False,
                                             fill_value=0.)
            for t in range(points.shape[0]):
                result[t, :, i] = interp(points[t])
        return result  # shape: (N_LST, N_sky, 2)

    def e_field_in_eq_coords(self, LSTs, sky_coords, freq_ind):
        """
        Compute the E-field in equatorial coordinates at given sky positions.

        Full pipeline: map equatorial coordinates to antenna frame, interpolate
        the E-field, and transform the field components back to equatorial basis.

        Parameters
        ----------
        LSTs : array_like
            Local Sidereal Times in seconds.
        sky_coords : ndarray, shape (N_sky, 2)
            Equatorial spherical coordinates (phi, theta) in radians.
        freq_ind : int
            Frequency channel index.

        Returns
        -------
        ndarray, shape (N_LST, N_sky, 2), dtype complex
            E-field in equatorial spherical basis (E_theta, E_phi).
            The radial component is discarded.
        """
        ant_coords, field_transform_operator = self.map_sky_coords_to_ant_coords(LSTs, sky_coords)
        interpolated_efield_ant = self.make_interp(ant_coords, freq_ind) # shape: (N_LST, N_sky, 2)
        del ant_coords
        result = np.einsum("tsij, tsj -> tsi", field_transform_operator, interpolated_efield_ant)
        return result[:, :, 1:]

    def generate_auto_beam_at_LSTs(self, LSTs, sky_coords, freq_ind, time_averaged = False):
        """
        Compute Full Stokes (I, Q, U, V) auto-correlation beam patterns.

        Computes the beam matrix via Pauli matrices:

            B_p(s) = Re[ E*(s) . sigma_p . E(s) ]

        where p indexes the four Stokes parameters.

        Einsum index conventions:
            t = LST, s = sky pixel, p = Stokes parameter,
            l/m = polarisation component (2x2 Pauli matrix indices).

        Parameters
        ----------
        LSTs : array_like
            Local Sidereal Times in seconds.
        sky_coords : ndarray, shape (N_sky, 2)
            Equatorial spherical coordinates (phi, theta) in radians.
        freq_ind : int
            Frequency channel index.
        time_averaged : bool, optional
            If True, average the E-field over all LSTs before computing the
            beam matrix. Result shape is (N_sky, 4).
            If False (default), compute per-LST beams. Result shape is
            (N_LST, N_sky, 4).

        Returns
        -------
        ndarray, shape (N_LST, N_sky, 4) or (N_sky, 4)
            Real-valued Stokes beam matrices. Last axis order: [I, Q, U, V].
        """
        if time_averaged:
            E_field = np.mean(self.e_field_in_eq_coords(LSTs, sky_coords, freq_ind), axis=0)
            B_matrix = np.einsum("sl, plm, sm -> sp",
                                 np.conjugate(E_field),
                                 pauli_array,
                                 E_field)
        else:
            E_field = self.e_field_in_eq_coords(LSTs, sky_coords, freq_ind)
            B_matrix = np.einsum("tsl, plm, tsm -> tsp",
                                 np.conjugate(E_field),
                                 pauli_array,
                                 E_field)
        return B_matrix.real


class EFieldBeamAngle():
    """
    Beam-angle-based E-field beam model using HEALPix interpolation.

    This class handles single-frequency E-field data and parameterises the
    antenna pointing by beam tilt angles (rather than LSTs). The E-field is
    stored on a HEALPix grid (nearest-neighbour assignment) for fast lookup
    during coordinate transformation.

    The transformation chain is similar to ``E_field``, but uses
    ``R_ant_h(beam_angles)`` instead of explicit alignment vectors and LSTs.

    Parameters
    ----------
    ant_coords : ndarray, shape (N_sky, 2)
        Antenna spherical coordinates (phi, theta) in radians, where
        N_sky = N_phi * N_theta.
    e_field_data : ndarray, shape (N_phi, N_theta, 2)
        Single-frequency complex E-field data. Last axis: (E_theta, E_phi).
    ant_lat : float
        Antenna latitude in radians.

    Attributes
    ----------
    Nside : int
        HEALPix Nside parameter (default 128, giving 196608 pixels).
    phi_array : ndarray
        Unique phi values from the antenna coordinate grid (radians).
    theta_array : ndarray
        Unique theta values from the antenna coordinate grid (radians).
    e_field_0_real, e_field_0_imag : ndarray, shape (N_pix,)
        Real and imaginary parts of E_theta on the HEALPix grid.
    e_field_1_real, e_field_1_imag : ndarray, shape (N_pix,)
        Real and imaginary parts of E_phi on the HEALPix grid.
    antenna_latitude : float
        Antenna latitude in radians.
    """

    def __init__(self, ant_coords, e_field_data, ant_lat):
        self.Nside = 128
        self.phi_array = np.unique(ant_coords[:, 0])
        self.theta_array = np.unique(ant_coords[:, 1])

        self.e_field_healpix(e_field_data)

        self.antenna_latitude = ant_lat

        print("The far-field object has been initialized!")

    def e_field_healpix(self, complex_field):
        """
        Convert the E-field from a regular (phi, theta) grid to HEALPix maps
        using nearest-neighbour assignment.

        For each HEALPix pixel, finds the closest (phi, theta) grid point and
        assigns its E-field value. Stores the real and imaginary parts of each
        polarisation component as separate HEALPix maps.

        Parameters
        ----------
        complex_field : ndarray, shape (N_phi, N_theta, 2)
            Complex E-field on the regular antenna grid.
        """
        nside = self.Nside
        npix = hp.nside2npix(nside)

        theta_vals = self.theta_array
        phi_vals = self.phi_array

        real_e_field_map = np.full(npix, hp.UNSEEN)
        imag_e_field_map = np.full(npix, hp.UNSEEN)
        real_e_field_map_1 = np.full(npix, hp.UNSEEN)
        imag_e_field_map_1 = np.full(npix, hp.UNSEEN)

        for idx in range(npix):
            h_theta, h_phi = hp.pix2ang(nside, idx)
            dphi = np.abs(phi_vals - h_phi)
            dtheta = np.abs(theta_vals - h_theta)
            closest_phi_idx = np.argmin(dphi)
            closest_theta_idx = np.argmin(dtheta)

            real_e_field_map[idx] = complex_field[closest_phi_idx, closest_theta_idx, 0].real
            imag_e_field_map[idx] = complex_field[closest_phi_idx, closest_theta_idx, 0].imag
            real_e_field_map_1[idx] = complex_field[closest_phi_idx, closest_theta_idx, 1].real
            imag_e_field_map_1[idx] = complex_field[closest_phi_idx, closest_theta_idx, 1].imag

        self.e_field_0_real = real_e_field_map
        self.e_field_0_imag = imag_e_field_map

        self.e_field_1_real = real_e_field_map_1
        self.e_field_1_imag = imag_e_field_map_1

        return

    def map_sky_coords_to_ant_coords(self, eq_sph_coords, beam_angles):
        """
        Map equatorial sky coordinates to antenna coordinates for given beam
        tilt angles.

        Similar to ``E_field.map_sky_coords_to_ant_coords`` but uses
        ``R_ant_h(beam_angles)`` for the antenna-to-horizontal rotation
        instead of explicit alignment vectors and LSTs. Sets LST = 0 internally.

        Einsum index conventions: d = beam angle, s = sky pixel, i/j/k/l = matrix.

        Parameters
        ----------
        eq_sph_coords : ndarray, shape (N_sky, 2)
            Equatorial spherical coordinates (phi, theta) in radians.
        beam_angles : array_like
            Beam tilt angles in radians.

        Returns
        -------
        ant_sph_coords : ndarray, shape (N_angles, N_sky, 2)
            Sky positions in antenna spherical coordinates.
        field_trans_mat : ndarray, shape (N_angles, N_sky, 3, 2)
            Field transformation operator.
        """
        R_ant2h, R_h2ant = R_ant_h(beam_angles, yy=False)
        eq_car_coords = coordinate_mapping_sph2car(eq_sph_coords)  # shape: (N_sky, 3)
        R_eq2h = rotation_matrix_eq2local(0., self.antenna_latitude)[0]
        ant_car_coords = np.einsum("dij,jk,sk -> dsi", R_h2ant, R_eq2h, eq_car_coords)
        ant_sph_coords = coordinate_mapping_car2sph(ant_car_coords)  # shape: (dim, N_sky, 2)
        R_sc = rotation_matrix_sph2car(ant_sph_coords)  # shape: (dim, N_sky, 3, 2)
        R_h2eq = rotation_matrix_local2eq(0., self.antenna_latitude)[0]
        R_cs_eq = rotation_matrix_car2sph(eq_sph_coords)  # shape: (N_sky, 3, 3)
        field_trans_mat = np.einsum("sij, jm, dmk, dskl -> dsil",
                                    R_cs_eq, R_h2eq, R_ant2h, R_sc) # shape: (dim, N_sky, 3, 2)
        return ant_sph_coords, field_trans_mat

    def make_interp(self, points):
        """
        Interpolate the E-field using RegularGridInterpolator.

        .. note::
            This method references ``self.field`` which is not set by this
            class's constructor. Use ``make_interp_2`` (HEALPix-based) instead,
            which is the method used by ``e_field_in_eq_coords``.

        Parameters
        ----------
        points : ndarray, shape (N_angles, N_sky, 2)
            Antenna spherical coordinates (phi, theta).

        Returns
        -------
        ndarray, shape (N_angles, N_sky, 2), dtype complex
            Interpolated E-field components.
        """
        result = np.zeros(shape=points.shape,  dtype=complex)
        for i in np.arange(2):
            interp = RegularGridInterpolator([self.phi_array, self.theta_array],
                                             self.field[:, :, i],
                                             bounds_error=False,
                                             method = "linear"
                                             #fill_value=0.
                                            )
            for d in range(points.shape[0]):
                result[d, :, i] = interp(points[d])
        return result  # shape: (dim, N_sky, 2)

    def make_interp_2(self, points):
        """
        Interpolate the E-field using HEALPix nearest-neighbour lookup.

        Converts the target (theta, phi) coordinates to HEALPix pixel indices
        via ``healpy.ang2pix`` and retrieves the stored E-field values.

        Parameters
        ----------
        points : ndarray, shape (N_angles, N_sky, 2)
            Antenna spherical coordinates (phi, theta) in radians.

        Returns
        -------
        ndarray, shape (N_angles, N_sky, 2), dtype complex
            Interpolated E-field components (E_theta, E_phi).
        """
        result = np.zeros(shape=points.shape,  dtype=complex)

        for t in range(points.shape[0]):
            target_theta, target_phi = points[t, :, 1], points[t, :, 0]
            # Convert (interp_theta, interp_phi) to HEALPix indices
            interp_indices = hp.ang2pix(self.Nside, target_theta, target_phi)
            result[t, :, 0] = self.e_field_0_real[interp_indices] + 1j*self.e_field_0_imag[interp_indices]
            result[t, :, 1] = self.e_field_1_real[interp_indices] + 1j*self.e_field_1_imag[interp_indices]
        return result  # shape: (N_LST, N_sky, 2)

    def e_field_in_eq_coords(self, sky_coords, beam_angles):
        """
        Compute the E-field in equatorial coordinates for given beam angles.

        Full pipeline: map equatorial coordinates to antenna frame, interpolate
        E-field via HEALPix lookup, and transform components to equatorial basis.

        Parameters
        ----------
        sky_coords : ndarray, shape (N_sky, 2)
            Equatorial spherical coordinates (phi, theta) in radians.
        beam_angles : array_like
            Beam tilt angles in radians.

        Returns
        -------
        ndarray, shape (N_angles, N_sky, 2), dtype complex
            E-field in equatorial spherical basis (E_theta, E_phi).
        """
        ant_coords, field_transform_operator = self.map_sky_coords_to_ant_coords(sky_coords, beam_angles)
        interpolated_efield_ant = self.make_interp_2(ant_coords) # shape: (dim, N_sky, 2)
        del ant_coords
        result = np.einsum("dsij, dsj -> dsi", field_transform_operator, interpolated_efield_ant)
        return result[:, :, 1:]

    def generate_auto_beam(self, sky_coords, beam_angles):
        """
        Compute Full Stokes (I, Q, U, V) auto-correlation beam patterns for
        a set of beam tilt angles.

        Computes: B_p(d, s) = Re[ E*(d, s) . sigma_p . E(d, s) ]

        Einsum index conventions:
            d = beam angle, s = sky pixel, p = Stokes parameter,
            l/m = polarisation component.

        Parameters
        ----------
        sky_coords : ndarray, shape (N_sky, 2)
            Equatorial spherical coordinates (phi, theta) in radians.
        beam_angles : array_like
            Beam tilt angles in radians.

        Returns
        -------
        ndarray, shape (N_angles, 4, N_sky)
            Real-valued Stokes beam maps. Axis 1 order: [I, Q, U, V].
        """
        E_field = self.e_field_in_eq_coords(sky_coords, beam_angles)
        B_matrix = np.einsum("dsl, plm, dsm -> dps",
                             np.conjugate(E_field),
                             pauli_array,
                             E_field)
        return B_matrix.real

    def generate_multipole_beams(self, sky_coords, beam_angles):
        """
        Compute Stokes beam maps and their spherical harmonic coefficients.

        Calls ``generate_auto_beam`` and then decomposes each Stokes map into
        spherical harmonics using ``healpy.map2alm``.

        Parameters
        ----------
        sky_coords : ndarray, shape (N_sky, 2)
            Equatorial spherical coordinates (phi, theta) in radians.
            N_sky should be compatible with a HEALPix map (12 * Nside^2 + 1,
            the extra pixel is discarded before ``map2alm``).
        beam_angles : array_like
            Beam tilt angles in radians.

        Returns
        -------
        B_maps : ndarray, shape (N_angles, 4, N_sky)
            Real-valued Stokes beam maps.
        B_alms : list of ndarray
            Spherical harmonic coefficients for each beam angle, computed by
            ``healpy.map2alm``.
        """
        B_maps_x = self.generate_auto_beam(sky_coords, beam_angles)
        B_matrix_x = []
        for d in range(B_maps_x.shape[0]):
            B_matrix_x.append(hp.sphtfunc.map2alm(B_maps_x[d][:-1]))
        return B_maps_x, B_matrix_x
