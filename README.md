# TIBEC — Time Integrated Beam pattern in Equatorial Coordinates

A Python library for transforming antenna far-field E-field patterns from
antenna-local coordinates into equatorial (RA/DEC) coordinates and computing
Full Stokes (I, Q, U, V) beam matrices.

## Installation

```bash
git clone <repo-url>
cd TIBEC
pip install -e .
```

Dependencies: `numpy`, `scipy`, `healpy`, `matplotlib`

## Quick Start

Compute Full Stokes beams from an E-field data file in one call:

```python
from tibec import compute_stokes_beams

stokes, sky_coords = compute_stokes_beams("data/HIRAX_201-Copy1.txt",
                                           n_phi=361, n_theta=181)
# stokes.shape = (49152, 4)  — columns: [I, Q, U, V]
```

Visualise with healpy:

```python
import healpy as hp
hp.mollview(stokes[:, 0], title="Stokes I")
```

For advanced usage with LST-dependent beams, see `examples/full_stokes_beam.ipynb`.

## Core Classes

### `E_field` — LST-based beam computation

For instruments with explicit antenna alignment (e.g., REACH). Supports
multi-frequency E-field data and time-dependent beam rotation.

```python
from tibec import E_field, load_efield_txt

coords, efield = load_efield_txt("data/REACH_Efield.txt", 73, 37, n_freq=26)

beam = E_field(coords, efield,
               alignment_x=[1, 0, 0],   # x-axis → South
               alignment_z=[0, 0, 1],   # z-axis → Zenith
               ant_lat=np.pi/2)

LSTs = np.linspace(0, 86400, 40)  # seconds
stokes = beam.generate_auto_beam_at_LSTs(LSTs, sky_coords, freq_ind=0)
# shape: (40, N_sky, 4)
```

### `EFieldBeamAngle` — Beam-angle-based computation

For instruments parameterised by beam tilt angles (e.g., HIRAX). Uses HEALPix
nearest-neighbour interpolation internally.

```python
from tibec import EFieldBeamAngle, load_efield_txt

coords, efield = load_efield_txt("data/HIRAX_201-Copy1.txt", 361, 181)

beam = EFieldBeamAngle(coords, efield, ant_lat=np.deg2rad(-30.7))

beam_angles = np.deg2rad(np.linspace(-10, 10, 5))
stokes = beam.generate_auto_beam(sky_coords, beam_angles)
# shape: (5, 4, N_sky)
```

## Data Format

E-field text files use 6-column format (FEKO/CST output):

```
phi(deg)  theta(deg)  Re(E_theta)  Im(E_theta)  Re(E_phi)  Im(E_phi)
```

Lines starting with `//` or `#` are comments. Multi-frequency files contain
repeated blocks of `n_phi * n_theta` rows.

## Coordinate Systems

TIBEC uses five coordinate systems (see `docs/TIBEC_documentation.pdf` for
full derivations):

| System | Notation | Description |
|--------|----------|-------------|
| Antenna spherical | (phi_a, theta_a) | Native far-field simulation coords |
| Antenna Cartesian | (x_a, y_a, z_a) | Derived from antenna spherical |
| Horizontal (local) | (x_h, y_h, z_h) | South, East, Zenith |
| Equatorial Cartesian | (x_e, y_e, z_e) | Celestial frame |
| Equatorial spherical | (phi_e, theta_e) | theta_e = pi/2 - DEC, phi_e = RA |

**Transformation chain:**

```
antenna spherical  →  antenna Cartesian  →  horizontal Cartesian  →  equatorial Cartesian  →  equatorial spherical
                  (Eq. 5-6)            (P, Eq. 3/7)           (R, Eq. 4/9)              (Eq. 11)
```

## Conventions

- All angles are in **radians**
- LST is in **seconds** (converted internally: phi = (LST / 3600) * pi / 12)
- Spherical coordinate arrays use **(phi, theta)** ordering
- E-field last axis: (E_theta, E_phi)
- Stokes order: I, Q, U, V (indices 0–3)

## Mathematical Documentation

Full derivations are in `docs/TIBEC_documentation.pdf`. A quick reference
is in `docs/math_reference.md`.

## License

MIT License. Copyright (c) 2022 Zheng Zhang.
