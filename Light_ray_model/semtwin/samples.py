"""Synthetic specimens made of points and small particles (bright on a dark substrate).

Each sample is a 2D array of the specimen signal over the scanned field of view, one value per
image pixel. Points are single bright pixels, so their image is (almost) the PSF itself.
"""
import numpy as np


def point_grid(n_px, spacing_px=24, brightness=1.0):
    """Regular grid of single-pixel points: shows the PSF at many field positions at once."""
    img = np.zeros((n_px, n_px))
    start = (n_px % spacing_px) // 2 + spacing_px // 2
    img[start::spacing_px, start::spacing_px] = brightness
    return img


def random_points(n_px, n_points=60, rng=None, margin_px=4):
    """Randomly placed single-pixel points with random brightness."""
    rng = np.random.default_rng(rng)
    img = np.zeros((n_px, n_px))
    iy, ix = rng.integers(margin_px, n_px - margin_px, (2, n_points))
    img[iy, ix] = rng.uniform(0.4, 1.0, n_points)
    return img


def particles(n_px, n_particles=40, radius_px=(0.6, 3.0), rng=None, supersample=4):
    """Small round particles (think tiny tin balls), drawn anti-aliased by supersampling."""
    rng = np.random.default_rng(rng)
    n_hi = n_px * supersample
    yy, xx = np.mgrid[0:n_hi, 0:n_hi] + 0.5
    hi = np.zeros((n_hi, n_hi))
    for _ in range(n_particles):
        r = rng.uniform(*radius_px) * supersample
        cx, cy = rng.uniform(4 * supersample, n_hi - 4 * supersample, 2)
        box = (slice(max(int(cy - r - 2), 0), int(cy + r + 2)), slice(max(int(cx - r - 2), 0), int(cx + r + 2)))
        inside = (xx[box] - cx) ** 2 + (yy[box] - cy) ** 2 <= r * r
        hi[box][inside] = rng.uniform(0.6, 1.0)
    return hi.reshape(n_px, supersample, n_px, supersample).mean(axis=(1, 3))


def make_samples(n_px, seed=0):
    """The default set of point-like specimens used in the figures and the database."""
    rng = np.random.default_rng(seed)
    return {
        "point grid": point_grid(n_px, spacing_px=max(n_px // 8, 8)),
        "random points": random_points(n_px, n_points=max(n_px // 3, 10), rng=rng),
        "particles": particles(n_px, n_particles=max(n_px // 5, 8), rng=rng),
    }
