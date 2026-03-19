"""
TIBEC: Time Integrated Beam pattern in Equatorial Coordinates.

A Python library for transforming antenna far-field E-field patterns from
antenna-local spherical coordinates into equatorial (RA/DEC) coordinates,
with support for time integration over Local Sidereal Time (LST).

See ``docs/TIBEC_documentation.pdf`` for the full mathematical derivations.
"""

from tibec.transformation import (
    coordinate_mapping_sph2car,
    coordinate_mapping_car2sph,
    rotation_matrix_local2eq,
    rotation_matrix_eq2local,
    rotation_matrix_sph2car,
    rotation_matrix_car2sph,
    pointing_matrix,
    R_ant_h,
)
from tibec.farfieldtransfer import (
    E_field,
    EFieldBeamAngle,
    Pauli_I, Pauli_Q, Pauli_U, Pauli_V,
    pauli_array,
)
from tibec.io import load_efield_txt, compute_stokes_beams

__version__ = "0.1.0"
