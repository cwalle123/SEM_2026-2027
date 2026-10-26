"""The SEM column digital twin (light-ray version).

Layout follows Fig. 1 of the report: gun -> condenser 1 -> condenser 2 -> aperture ->
scan coils -> stigmator -> objective -> specimen. Glass lenses stand in for the magnetic lenses,
the "strength" of each lens stands in for its coil current.

Two things you work with:
* `SEMSettings`: the knobs of the machine (what an operator / misalignment changes).
* `SEMColumn`:   the fixed hardware. It traces rays for a given set of knobs and returns the probe.
"""
from dataclasses import dataclass, field, replace, asdict

import numpy as np
from scipy.stats import norm, qmc

from .optics import Rays, Lens, Aperture, Deflector, Stigmator


@dataclass
class SEMSettings:
    # gun (light source)
    source_radius_um: float = 2.0      # finite source size -> geometric blur (like d50,geo)
    wavelength_nm: float = 550.0
    bandwidth_nm: float = 0.0          # 1-sigma spectral spread -> chromatic aberration (like Cc)
    # condenser lenses
    c1_strength: float = 1.0
    c2_strength: float = 1.0
    # final aperture
    aperture_radius_mm: float = 1.0
    aperture_shift_x_mm: float = 0.0
    aperture_shift_y_mm: float = 0.0
    # scan coils: where on the specimen the probe is sent
    scan_x_mm: float = 0.0
    scan_y_mm: float = 0.0
    # stigmator (quadrupole) [1/mm]
    stig_x: float = 0.0
    stig_y: float = 0.0
    # objective lens: strength (focus) and misalignment
    obj_strength: float = 1.0
    obj_tilt_x_mrad: float = 0.0
    obj_tilt_y_mrad: float = 0.0
    obj_shift_x_mm: float = 0.0
    obj_shift_y_mm: float = 0.0
    # specimen stage height: + moves the specimen further from the objective
    defocus_um: float = 0.0

    def replace(self, **changes):
        return replace(self, **changes)

    def as_dict(self):
        return asdict(self)


@dataclass
class LensSpec:
    z: float
    r1: float
    r2: float
    thickness: float
    semi_diameter: float


@dataclass
class ColumnGeometry:
    """Fixed hardware positions [mm]. C1 images the source to a crossover, C2 roughly
    collimates it, the objective focuses the beam onto the specimen."""
    source_z: float = 0.0
    c1: LensSpec = field(default_factory=lambda: LensSpec(30.0, 15.6, -15.6, 2.5, 3.0))
    c2: LensSpec = field(default_factory=lambda: LensSpec(100.0, 41.5, -41.5, 2.0, 4.0))
    aperture_z: float = 120.0
    aperture_plate_radius: float = 6.0
    scan1_z: float = 140.0
    scan2_z: float = 160.0
    stigmator_z: float = 172.0
    objective: LensSpec = field(default_factory=lambda: LensSpec(190.0, 15.6, -15.6, 3.0, 5.0))


@dataclass
class Probe:
    """Result of one trace: where the rays cross a set of planes around the specimen."""
    settings: SEMSettings
    planes_um: np.ndarray            # plane offsets from the specimen surface [um] (+ = deeper)
    plane_z: np.ndarray              # absolute z of those planes [mm]
    hits: list                       # per plane: (x, y) arrays [mm] of the arriving rays
    n_launched: int                  # rays launched from the source
    rays: Rays                       # the bundle after the objective (for drawing)

    @property
    def target(self):
        return np.array([self.settings.scan_x_mm, self.settings.scan_y_mm])

    def plane_index(self, offset_um=0.0):
        return int(np.argmin(np.abs(self.planes_um - offset_um)))


class SEMColumn:
    def __init__(self, geometry=None, n_rays=2 ** 17):
        self.geometry = geometry or ColumnGeometry()
        self.n_rays = n_rays
        self._specimen_z = None
        self._scan_gain = None

    # ---------------------------------------------------------------- elements
    def elements(self, s: SEMSettings):
        g = self.geometry
        kick1, kick2 = self._scan_kicks(s)
        mk = lambda name, spec, k, sh, ti: Lens(name, spec.z, spec.r1, spec.r2, spec.thickness,
                                                 spec.semi_diameter, k, sh, ti)
        return [
            mk("Condenser 1", g.c1, s.c1_strength, (0, 0), (0, 0)),
            mk("Condenser 2", g.c2, s.c2_strength, (0, 0), (0, 0)),
            Aperture("Aperture", g.aperture_z, s.aperture_radius_mm,
                     (s.aperture_shift_x_mm, s.aperture_shift_y_mm), g.aperture_plate_radius),
            Deflector("Scan coil 1", g.scan1_z, kick1),
            Deflector("Scan coil 2", g.scan2_z, kick2),
            Stigmator("Stigmator", g.stigmator_z, (s.stig_x, s.stig_y)),
            mk("Objective", g.objective, s.obj_strength, (s.obj_shift_x_mm, s.obj_shift_y_mm),
               (s.obj_tilt_x_mrad * 1e-3, s.obj_tilt_y_mrad * 1e-3)),
        ]

    def _scan_kicks(self, s):
        """Double deflection: tilt the beam by theta while pivoting it about the objective centre,
        so it stays centred in the lens. theta is chosen from the (nominal) scan calibration."""
        if s.scan_x_mm == 0 and s.scan_y_mm == 0:
            return (0.0, 0.0), (0.0, 0.0)
        g = self.geometry
        gain = self.scan_gain
        theta = np.array([s.scan_x_mm, s.scan_y_mm]) / gain
        a = -theta * (g.objective.z - g.scan2_z) / (g.scan2_z - g.scan1_z)
        return tuple(a), tuple(theta - a)

    # ---------------------------------------------------------------- source
    def _source_na(self, s):
        """Emission half-angle (slope) that just over-fills the aperture, so few rays are wasted."""
        nominal = s.replace(scan_x_mm=0, scan_y_mm=0)
        c1, c2 = self.elements(nominal)[:2]
        rays = Rays([[0, 0, 0], [s.source_radius_um * 1e-3, 0, 0]], [[1e-4, 0, 1], [0, 0, 1]], s.wavelength_nm)
        c1.apply(rays)
        c2.apply(rays)
        rays.propagate_to_z(self.geometry.aperture_z)
        h_per_slope = abs(rays.pos[0, 0]) / 1e-4
        h_source = abs(rays.pos[1, 0])
        reach = s.aperture_radius_mm + np.hypot(s.aperture_shift_x_mm, s.aperture_shift_y_mm) + h_source
        return 1.1 * reach / h_per_slope

    def emit(self, s, n_rays=None, seed=0, record=False):
        """Launch rays from the source. A scrambled Sobol sequence gives smooth PSFs with few rays."""
        n = n_rays or self.n_rays
        m = int(np.ceil(np.log2(max(n, 2))))
        u = qmc.Sobol(d=5, scramble=True, seed=seed).random_base2(m)[:n]
        na = self._source_na(s)
        r, phi = na * np.sqrt(u[:, 0]), 2 * np.pi * u[:, 1]
        rs, phs = s.source_radius_um * 1e-3 * np.sqrt(u[:, 2]), 2 * np.pi * u[:, 3]
        pos = np.stack([rs * np.cos(phs), rs * np.sin(phs), np.full(n, self.geometry.source_z)], 1)
        dirs = np.stack([r * np.cos(phi), r * np.sin(phi), np.ones(n)], 1)
        wl = s.wavelength_nm + s.bandwidth_nm * norm.ppf(np.clip(u[:, 4], 1e-6, 1 - 1e-6))
        return Rays(pos, dirs, wl, record=record)

    # ---------------------------------------------------------------- tracing
    def trace(self, s, n_rays=None, seed=0, record=False, rays=None):
        """Trace from the source to just after the objective. Returns the Rays bundle."""
        rays = rays if rays is not None else self.emit(s, n_rays, seed, record)
        for el in self.elements(s):
            el.apply(rays)
        return rays

    def specimen_plane(self, s, offset_um=0.0):
        return self.specimen_z + (s.defocus_um + offset_um) * 1e-3

    def probe(self, s, planes_um=(0.0,), n_rays=None, seed=0):
        """Trace the beam and intersect it with planes at the given offsets from the specimen."""
        rays = self.trace(s, n_rays, seed)
        planes_um = np.atleast_1d(np.asarray(planes_um, float))
        plane_z = np.array([self.specimen_plane(s, o) for o in planes_um])
        hits = [rays.at_z(z) for z in plane_z]
        return Probe(s, planes_um, plane_z, hits, len(rays), rays)

    # ---------------------------------------------------------------- calibration
    @property
    def specimen_z(self):
        """Working distance: the specimen sits at the best (minimum-RMS) focus of the nominal machine.
        This is what an operator does once when focusing; afterwards every knob change can defocus."""
        if self._specimen_z is None:
            rays = self.trace(SEMSettings(source_radius_um=0.0), n_rays=2 ** 14)
            p, d = rays.pos[rays.alive], rays.dir[rays.alive]
            u, v = d[:, 0] / d[:, 2], d[:, 1] / d[:, 2]
            z0 = p[:, 2].max()
            x0, y0 = p[:, 0] + u * (z0 - p[:, 2]), p[:, 1] + v * (z0 - p[:, 2])
            dx, dy, du, dv = x0 - x0.mean(), y0 - y0.mean(), u - u.mean(), v - v.mean()
            # RMS spot radius is quadratic in z, so its minimum is found exactly
            self._specimen_z = z0 - np.sum(dx * du + dy * dv) / np.sum(du * du + dv * dv)
        return self._specimen_z

    @property
    def scan_gain(self):
        """mm of probe displacement on the specimen per radian of beam tilt at the objective."""
        if self._scan_gain is None:
            g = self.geometry
            theta = 1e-3
            a = -theta * (g.objective.z - g.scan2_z) / (g.scan2_z - g.scan1_z)
            s = SEMSettings(source_radius_um=0.0)
            els = self.elements(s)
            els[3].kick_xy, els[4].kick_xy = (a, 0.0), (theta - a, 0.0)
            rays = Rays([[0, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 1]], s.wavelength_nm)
            for el in els:
                el.apply(rays)
            x, _ = rays.at_z(self.specimen_z)
            self._scan_gain = x[0] / theta
        return self._scan_gain

    def demo_rays(self, s, n_rings=3, n_spokes=12, include_blocked=True):
        """A small, regular fan of rays from the source centre for drawing: rings that pass the
        aperture plus one ring that gets blocked. With a bandwidth, the fan is repeated at
        -2, 0, +2 sigma in wavelength. Returns (rays, ring fraction per ray)."""
        na_pass = self._source_na(s) / 1.1
        fracs = list(np.linspace(1 / n_rings, 0.97, n_rings)) + ([1.12] if include_blocked else [])
        dirs, ring = [[0, 0, 1]], [0.0]
        for f in fracs:
            for k in range(n_spokes):
                phi = 2 * np.pi * k / n_spokes
                dirs.append([f * na_pass * np.cos(phi), f * na_pass * np.sin(phi), 1])
                ring.append(f)
        sigmas = (-2, 0, 2) if s.bandwidth_nm > 0 else (0,)
        dirs = np.tile(dirs, (len(sigmas), 1))
        wl = np.repeat([s.wavelength_nm + k * s.bandwidth_nm for k in sigmas], len(ring))
        pos = np.tile([0.0, 0.0, self.geometry.source_z], (len(dirs), 1))
        rays = Rays(pos, dirs, wl, record=True)
        return self.trace(s, rays=rays), np.tile(ring, len(sigmas))
