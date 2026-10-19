"""Scanning image formation.

The probe is raster-scanned over the specimen; each image pixel is the specimen signal weighted
by the probe sitting at that scan position:  I(X) = sum_r S(r) * P_X(r - X).
That is a cross-correlation with the probe (a convolution with the flipped probe).

Because the probe changes across the field (off-axis aberrations, distortion), the PSF is
computed at a grid of scan positions and blended bilinearly between them.
"""
import numpy as np
from scipy.signal import correlate

from .psf import psf_image


def field_nodes(fov_mm, n_px, n_field):
    px = fov_mm / n_px
    half = fov_mm / 2 - px / 2
    return np.linspace(-half, half, n_field) if n_field > 1 else np.zeros(1)


def _tent_weights(coords, nodes):
    if len(nodes) == 1:
        return np.ones((1, len(coords)))
    step = nodes[1] - nodes[0]
    return np.clip(1 - np.abs(coords[None, :] - nodes[:, None]) / step, 0, 1)


def scan_kernels(column, settings, fov_mm, n_px, n_field=3, kernel_px=41, n_rays=2 ** 15):
    """Probe kernels on the image pixel grid at an n_field x n_field grid of scan positions.

    Kernels are centred on the intended scan position plus the mean probe offset over the field.
    So the uniform image shift a misaligned lens causes is removed (an operator re-centres the
    image), while shifts that vary across the field (distortion) stay visible in the image."""
    window_um = kernel_px * fov_mm / n_px * 1e3
    nodes = field_nodes(fov_mm, n_px, n_field)
    probes = {}
    for iy, Y in enumerate(nodes):
        for ix, X in enumerate(nodes):
            s = settings.replace(scan_x_mm=settings.scan_x_mm + X, scan_y_mm=settings.scan_y_mm + Y)
            probes[iy, ix] = column.probe(s, (0.0,), n_rays=n_rays)
    offsets = [np.mean(pr.hits[0], axis=1) - pr.target for pr in probes.values() if len(pr.hits[0][0])]
    offset = np.mean(offsets, axis=0) if offsets else np.zeros(2)
    kernels = np.zeros((len(nodes), len(nodes), kernel_px, kernel_px))
    for (iy, ix), pr in probes.items():
        x, y = pr.hits[0]
        kernels[iy, ix] = psf_image(x, y, pr.target + offset, window_um, kernel_px, n_launched=pr.n_launched)
    # normalise to the on-axis transmitted current so dimmer probes (vignetting) stay dimmer
    centre = kernels[len(nodes) // 2, len(nodes) // 2].sum()
    return kernels / max(centre, 1e-12), nodes


def scan_image(column, settings, sample, fov_mm, n_field=3, kernel_px=41, n_rays=2 ** 15,
               kernels=None, dose=None, rng=None):
    """Simulated SEM image of `sample` (2D array over the field of view, one value per pixel).

    dose: if given, mean detected counts for a pixel of signal 1 -> adds Poisson shot noise.
    kernels: reuse precomputed (kernels, nodes) from scan_kernels for several samples.
    """
    n_px = sample.shape[0]
    if kernels is None:
        kernels = scan_kernels(column, settings, fov_mm, n_px, n_field, kernel_px, n_rays)
    kernels, nodes = kernels
    coords = (np.arange(n_px) - (n_px - 1) / 2) * fov_mm / n_px
    w = _tent_weights(coords, nodes)
    image = np.zeros_like(sample, dtype=float)
    for iy in range(len(nodes)):
        for ix in range(len(nodes)):
            blurred = correlate(sample, kernels[iy, ix], mode="same", method="fft")
            image += w[iy][:, None] * w[ix][None, :] * blurred
    image = np.clip(image, 0, None)
    if dose is not None:
        rng = np.random.default_rng(rng)
        image = rng.poisson(image * dose) / dose
    return image
