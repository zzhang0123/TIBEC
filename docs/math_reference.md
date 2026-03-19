# TIBEC Mathematical Reference

Quick reference for the coordinate systems, transformations, and beam
computation used in TIBEC. For full derivations, see `TIBEC_documentation.pdf`.

## Coordinate Systems

### 1. Antenna Spherical (phi_a, theta_a)

The default coordinate system from the far-field simulation output.
Used to describe the linearly scaled far-field patterns.

### 2. Antenna Cartesian (x_a, y_a, z_a)

Derived from antenna spherical via standard mapping (r = 1):

    x_a = sin(theta_a) cos(phi_a)
    y_a = sin(theta_a) sin(phi_a)
    z_a = cos(theta_a)

- z_a: parallel to the direction from origin to pointing centre
- x_a: parallel to the radial direction at (theta = pi/2, phi = 0)
- y_a: by right-hand convention

### 3. Horizontal (Local) Cartesian (x_h, y_h, z_h)

Defined by cardinal directions relative to the antenna:

- x_h (South): identical to **S-hat**, the true South direction
- y_h (East): identical to **E-hat**, the true East direction
- z_h (Zenith): identical to **Z-hat**, the direction from antenna to zenith

### 4. Equatorial Cartesian (x_e, y_e, z_e)

Based on the celestial coordinate system:

- z_e: parallel to direction to the north celestial pole
- x_e: parallel to the radial direction at (theta_e = pi/2, phi_e = 0)
- y_e: by right-hand convention

### 5. Equatorial Spherical (phi_e, theta_e)

Same as the (RA, DEC) system but in radians:

    theta_e = pi/2 - DEC
    phi_e   = RA

## Key Matrices

### Pointing Matrix P (Eq. 3)

Maps antenna Cartesian to horizontal Cartesian:

    [x_h, y_h, z_h]^T = P @ [x_a, y_a, z_a]^T

Columns of P are the unit antenna axes [x_hat, y_hat, z_hat] expressed in
horizontal coordinates. P is orthogonal, so P^T maps horizontal to antenna.

For an antenna with x-axis = South and z-axis = Zenith:

    x_Axis = [1, 0, 0],  z_Axis = [0, 0, 1]  →  P = I (identity)

### Rotation Matrix R_{eq-h} (Eq. 4)

Rotation between equatorial and horizontal Cartesian frames:

    R_{eq-h}(theta, phi) = | cos(theta)cos(phi)  cos(theta)sin(phi)  -sin(theta) |
                           | -sin(phi)            cos(phi)             0           |
                           | sin(theta)cos(phi)  sin(theta)sin(phi)   cos(theta)  |

where:

    theta = pi/2 - latitude
    phi   = (LST / 3600) * (pi / 12)

R is orthogonal: R_{h-eq} = R_{eq-h}^T.

### Basis Rotation R_{s-c} (Eq. 6)

Transforms E-field components from spherical to Cartesian basis:

    [E_x, E_y, E_z]^T = R_{s-c}(theta, phi) @ [0, E_theta, E_phi]^T

    R_{s-c} = | sin(theta)cos(phi)  cos(theta)cos(phi)  -sin(phi) |
              | sin(theta)sin(phi)  cos(theta)sin(phi)   cos(phi) |
              | cos(theta)          -sin(theta)           0        |

Only columns 2-3 are used (column 1 corresponds to E_r = 0 in the far field).

## Transformation Chain

### Coordinates (equatorial → antenna)

    equatorial spherical → equatorial Cartesian → horizontal Cartesian → antenna Cartesian → antenna spherical
                                                 (R_{eq-h})            (P^T)

### E-field components (antenna → equatorial)

    E(antenna spherical basis) → E(antenna Cartesian) → E(horizontal Cartesian) → E(equatorial Cartesian) → E(equatorial spherical basis)
                                (R_{s-c, ant})         (P = R_{ant-h})           (R_{h-eq})                ([R_{s-c, eq}]^T)

Combined as a single einsum operation:

    E_eq = R_{c-s,eq} @ R_{h-eq} @ P @ R_{s-c,ant} @ E_ant

## Stokes Beam Computation

The Full Stokes beam matrix is computed using the Pauli matrices:

    B_p(s) = Re[ E*(s) . sigma_p . E(s) ]

where E = (E_theta, E_phi)^T is the 2-component complex E-field vector and:

    sigma_I = 1/2 |1  0|    sigma_Q = 1/2 | 1  0|
                  |0  1|                   | 0 -1|

    sigma_U = 1/2 |0  1|    sigma_V = 1/2 | 0 -i|
                  |1  0|                   | i  0|

## Einsum Index Conventions

Used throughout the code:

| Index | Meaning |
|-------|---------|
| `t` | LST (time) |
| `s` | sky pixel |
| `d` | beam angle dimension |
| `p` | Stokes parameter (I, Q, U, V) |
| `i, j, k, l, m` | matrix indices (3x3 or 2x2) |
