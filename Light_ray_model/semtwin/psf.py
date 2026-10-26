"""Turning ray hits on a plane into a PSF image, plus simple spot metrics."""
import numpy as np


def psf_image(x_mm, y_mm, center_mm, window_um, npix, n_launched=None):
    """Bin ray hits into an npix x npix image covering window_um around center_mm.

    Cloud-in-cell deposition (each ray shared over its 4 nearest pixels) keeps the PSF smooth.
    Array layout is [y, x] with y increasing with row index (plot with origin='lower').
    Normalised so the sum is the fraction of launched rays that land in the window
    (or sums to 1 over all arriving rays when n_launched is None).
    """
    px = window_um / npix
    gx = ((np.asarray(x_mm) - center_mm[0]) * 1e3 + window_um / 2) / px - 0.5
    gy = ((np.asarray(y_mm) - center_mm[1]) * 1e3 + window_um / 2) / px - 0.5
    ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - ix, gy - iy
    img = np.zeros(npix * npix)
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        jx, jy = ix + dx, iy + dy
        ok = (jx >= 0) & (jx < npix) & (jy >= 0) & (jy < npix)
        img += np.bincount(jy[ok] * npix + jx[ok], weights=w[ok], minlength=npix * npix)
    total = n_launched if n_launched is not None else max(len(gx), 1)
    return img.reshape(npix, npix) / total


def spot_metrics(x_mm, y_mm):
    """Centroid [mm], RMS radius [um], d50 (diameter holding 50% of the rays) [um], max radius [um]."""
    x, y = np.asarray(x_mm), np.asarray(y_mm)
    if len(x) == 0:
        return dict(cx_mm=np.nan, cy_mm=np.nan, rms_um=np.nan, d50_um=np.nan, rmax_um=np.nan)
    cx, cy = x.mean(), y.mean()
    r = np.hypot(x - cx, y - cy) * 1e3
    return dict(cx_mm=cx, cy_mm=cy, rms_um=float(np.sqrt(np.mean(r ** 2))),
                d50_um=float(2 * np.median(r)), rmax_um=float(r.max()))


def probe_psfs(probe, window_um=64.0, npix=96, center="focus_centroid"):
    """PSF image for every plane of a Probe, all on the same grid.

    center: 'focus_centroid' (centroid at the specimen plane, keeps lateral drift between planes
            visible), 'target' (the intended scan position, keeps shifts/distortion visible),
            or an explicit (x, y) in mm.
    """
    if isinstance(center, str):
        if center == "target":
            c = probe.target
        else:
            x, y = probe.hits[probe.plane_index(0.0)]
            c = np.array([x.mean(), y.mean()]) if len(x) else probe.target
    else:
        c = np.asarray(center, float)
    return np.stack([psf_image(x, y, c, window_um, npix) for x, y in probe.hits]), c
