"""Geometric (ray) optics in 3D: ray bundles, glass, and the optical elements of the column.

Conventions
-----------
* Lengths in mm, wavelengths in nm.
* The beam travels in +z (source at z = 0, specimen at the bottom, largest z).
* Every element acts on a `Rays` bundle in place. Rays that hit a lens rim, the aperture
  plate or undergo total internal reflection are marked dead and ignored afterwards.
"""
import numpy as np


def glass_index(wavelength_nm, A=1.5046, B=4200.0):
    """Refractive index of a BK7-like glass (Cauchy fit). Dispersion gives chromatic aberration,
    the light analogue of the electron energy spread (Cc)."""
    return A + B / np.asarray(wavelength_nm, float) ** 2


def tilt_matrix(tilt_x, tilt_y):
    """Rotation that tips an element's local z axis towards +x by tilt_x and towards +y by tilt_y [rad]."""
    cx, sx = np.cos(tilt_x), np.sin(tilt_x)
    cy, sy = np.cos(tilt_y), np.sin(tilt_y)
    rot_y = np.array([[cx, 0, sx], [0, 1, 0], [-sx, 0, cx]])
    rot_x = np.array([[1, 0, 0], [0, cy, sy], [0, -sy, cy]])
    return rot_y @ rot_x


class Rays:
    """A bundle of rays: positions (N,3) [mm], unit directions (N,3), wavelengths (N,) [nm].

    With record=True every element appends a snapshot of the positions, so the full ray
    paths can be drawn afterwards (`path`, `path_alive`).
    """

    def __init__(self, pos, dirs, wavelength, record=False):
        self.pos = np.array(pos, float)
        dirs = np.array(dirs, float)
        self.dir = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        self.wl = np.broadcast_to(np.asarray(wavelength, float), (len(self.pos),)).copy()
        self.alive = np.ones(len(self.pos), bool)
        self.record = record
        self.path = [self.pos.copy()] if record else None
        self.path_alive = [self.alive.copy()] if record else None

    def __len__(self):
        return len(self.pos)

    def snapshot(self):
        if self.record:
            self.path.append(self.pos.copy())
            self.path_alive.append(self.alive.copy())

    def propagate_to_z(self, z):
        s = (z - self.pos[:, 2]) / self.dir[:, 2]
        self.pos = self.pos + s[:, None] * self.dir

    def at_z(self, z):
        """(x, y) of the living rays where they cross the plane z, without moving the bundle."""
        p, d = self.pos[self.alive], self.dir[self.alive]
        s = (z - p[:, 2]) / d[:, 2]
        return p[:, 0] + s * d[:, 0], p[:, 1] + s * d[:, 1]

    def kick(self, du, dv):
        """Add du, dv to the ray slopes dx/dz, dy/dz (what a deflector or multipole does)."""
        u = self.dir[:, 0] / self.dir[:, 2] + du
        v = self.dir[:, 1] / self.dir[:, 2] + dv
        d = np.stack([u, v, np.ones_like(u)], axis=1)
        self.dir = d / np.linalg.norm(d, axis=1, keepdims=True)


def _refract_sphere(p, d, alive, z_vertex, c, n1, n2, semi_diameter):
    """Intersect rays with a spherical surface (vertex on the local axis at z_vertex, curvature c)
    and refract them with the vector form of Snell's law. Returns new p, d and the survivors."""
    q = p - np.array([0.0, 0.0, z_vertex])
    # sphere through the vertex: c*(x^2+y^2+z^2) - 2z = 0  ->  c s^2 + 2 b s + cc = 0
    b = c * np.einsum("ij,ij->i", q, d) - d[:, 2]
    cc = c * np.einsum("ij,ij->i", q, q) - 2 * q[:, 2]
    disc = b * b - c * cc
    denom = -b + np.sqrt(np.maximum(disc, 0.0))
    ok = alive & (disc >= 0) & (np.abs(denom) > 1e-15)
    s = np.where(ok, cc / np.where(ok, denom, 1.0), 0.0)  # stable root, nearest to the vertex
    q = q + s[:, None] * d
    ok &= q[:, 0] ** 2 + q[:, 1] ** 2 <= semi_diameter ** 2

    normal = np.stack([c * q[:, 0], c * q[:, 1], c * q[:, 2] - 1.0], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    cos_i = -np.einsum("ij,ij->i", normal, d)
    normal[cos_i < 0] *= -1
    cos_i = np.abs(cos_i)
    eta = n1 / n2
    k = 1 - eta ** 2 * (1 - cos_i ** 2)
    ok &= k >= 0  # total internal reflection -> lost
    t = eta[:, None] * d + (eta * cos_i - np.sqrt(np.maximum(k, 0.0)))[:, None] * normal
    d_new = np.where(ok[:, None], t, d)
    return q + np.array([0.0, 0.0, z_vertex]), d_new, ok


class Element:
    name = "element"
    z = 0.0

    def apply(self, rays):
        raise NotImplementedError


class Lens(Element):
    """Thick glass lens with two spherical faces.

    z            : axial position of the lens centre [mm]
    r1, r2       : radii of the entrance / exit faces [mm] (np.inf = flat, sign convention: centre
                   of curvature at +R from the vertex, so a biconvex lens has r1 > 0, r2 < 0)
    strength     : multiplies both curvatures. The glass stand-in for an electron lens coil current.
    shift, tilt  : misalignment of the lens: decentre (x, y) [mm] and tilt (x, y) [rad].
    """

    def __init__(self, name, z, r1, r2, thickness, semi_diameter, strength=1.0,
                 shift=(0.0, 0.0), tilt=(0.0, 0.0)):
        self.name, self.z = name, z
        self.r1, self.r2 = r1, r2
        self.thickness, self.semi_diameter = thickness, semi_diameter
        self.strength = strength
        self.shift, self.tilt = shift, tilt

    @property
    def curvatures(self):
        return tuple(0.0 if np.isinf(r) else self.strength / r for r in (self.r1, self.r2))

    @property
    def origin(self):
        return np.array([self.shift[0], self.shift[1], self.z])

    @property
    def rotation(self):
        return tilt_matrix(*self.tilt)

    def apply(self, rays):
        o, R = self.origin, self.rotation
        p = (rays.pos - o) @ R
        d = rays.dir @ R
        n_glass = glass_index(rays.wl)
        ones = np.ones_like(n_glass)
        c1, c2 = self.curvatures
        h = self.thickness / 2
        for z_v, c, n1, n2 in ((-h, c1, ones, n_glass), (h, c2, n_glass, ones)):
            p, d, ok = _refract_sphere(p, d, rays.alive, z_v, c, n1, n2, self.semi_diameter)
            rays.alive &= ok
            rays.pos = p @ R.T + o
            rays.snapshot()
        rays.dir = d @ R.T


class Aperture(Element):
    """Circular hole in an opaque plate. It sets the beam convergence angle (and current)."""

    def __init__(self, name, z, radius, shift=(0.0, 0.0), outer_radius=6.0):
        self.name, self.z = name, z
        self.radius, self.shift, self.outer_radius = radius, shift, outer_radius

    def apply(self, rays):
        rays.propagate_to_z(self.z)
        r2 = (rays.pos[:, 0] - self.shift[0]) ** 2 + (rays.pos[:, 1] - self.shift[1]) ** 2
        rays.alive &= r2 <= self.radius ** 2
        rays.snapshot()


class Deflector(Element):
    """Scan coil: a uniform field that tilts every ray by the same angle (kick in slope)."""

    def __init__(self, name, z, kick=(0.0, 0.0)):
        self.name, self.z, self.kick_xy = name, z, kick

    def apply(self, rays):
        rays.propagate_to_z(self.z)
        rays.kick(*self.kick_xy)
        rays.snapshot()


class Stigmator(Element):
    """Thin quadrupole (two-fold astigmatism corrector). strength = (s_x, s_y) [1/mm]:
    s_x focuses in x and defocuses in y, s_y does the same along the diagonals."""

    def __init__(self, name, z, strength=(0.0, 0.0)):
        self.name, self.z, self.strength = name, z, strength

    def apply(self, rays):
        rays.propagate_to_z(self.z)
        sx, sy = self.strength
        x, y = rays.pos[:, 0], rays.pos[:, 1]
        rays.kick(-(sx * x + sy * y), -(sy * x - sx * y))
        rays.snapshot()
