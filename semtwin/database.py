"""Build a database that maps machine settings -> probe (through-focus PSFs) -> images of samples.

Each entry is one random machine configuration (Latin hypercube over PARAM_RANGES). For it we store
* the settings themselves,
* the PSF at a through-focus series of planes (under / in / over focus, like the report's method),
* spot metrics (centroid shift, RMS radius, d50) per plane,
* the simulated scan image of every sample (with field-dependent PSF).
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from .column import SEMSettings
from .imaging import scan_image, scan_kernels
from .psf import probe_psfs, spot_metrics
from .samples import make_samples

# knob -> (low, high). Everything not listed stays at its SEMSettings default.
PARAM_RANGES = {
    "defocus_um": (-80.0, 80.0),
    "aperture_radius_mm": (0.4, 1.6),
    "aperture_shift_x_mm": (-0.3, 0.3),
    "aperture_shift_y_mm": (-0.3, 0.3),
    "stig_x": (-4e-4, 4e-4),
    "stig_y": (-4e-4, 4e-4),
    "obj_tilt_x_mrad": (-50.0, 50.0),
    "obj_tilt_y_mrad": (-50.0, 50.0),
    "obj_shift_x_mm": (-0.3, 0.3),
    "obj_shift_y_mm": (-0.3, 0.3),
    "c2_strength": (0.98, 1.02),
    "bandwidth_nm": (0.0, 30.0),
    "source_radius_um": (0.0, 6.0),
}


def sample_settings(n, ranges=PARAM_RANGES, seed=0, base=None):
    base = base or SEMSettings()
    names = list(ranges)
    lo, hi = np.array([ranges[k] for k in names]).T
    u = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    values = lo + u * (hi - lo)
    return names, values, [base.replace(**dict(zip(names, row))) for row in values]


def build_database(column, n_configs=300, out_dir="output/database", ranges=PARAM_RANGES, seed=0,
                   planes_um=(-50.0, 0.0, 50.0), psf_window_um=64.0, psf_px=64,
                   fov_mm=0.3, image_px=128, n_field=3, kernel_px=33, dose=None, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, values, settings = sample_settings(n_configs, ranges, seed)
    samples = make_samples(image_px, seed=seed)
    sample_names = list(samples)
    sample_stack = np.stack([samples[k] for k in sample_names])

    psfs = np.zeros((n_configs, len(planes_um), psf_px, psf_px), np.float32)
    images = np.zeros((n_configs, len(sample_names), image_px, image_px), np.float16)
    metric_names = ["shift_x_um", "shift_y_um"] + [f"{m}_{p:+.0f}um" for p in planes_um
                                                    for m in ("rms_um", "d50_um")]
    metrics = np.zeros((n_configs, len(metric_names)), np.float32)

    t0 = time.perf_counter()
    for i, s in enumerate(settings):
        pr = column.probe(s, planes_um)
        stack, centre = probe_psfs(pr, psf_window_um, psf_px)
        psfs[i] = stack
        row = list((centre - pr.target) * 1e3)
        for x, y in pr.hits:
            m = spot_metrics(x, y)
            row += [m["rms_um"], m["d50_um"]]
        metrics[i] = row
        k = scan_kernels(column, s, fov_mm, image_px, n_field, kernel_px)
        for j, name in enumerate(sample_names):
            images[i, j] = scan_image(column, s, samples[name], fov_mm, kernels=k, dose=dose, rng=seed + i)
        if verbose and (i + 1) % max(n_configs // 20, 1) == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{n_configs} configs  ({el:.0f}s elapsed, ~{el / (i + 1) * (n_configs - i - 1):.0f}s left)")

    meta = dict(param_names=names, param_ranges={k: ranges[k] for k in names}, planes_um=list(planes_um),
                psf_window_um=psf_window_um, psf_px=psf_px, fov_mm=fov_mm, image_px=image_px,
                n_field=n_field, kernel_px=kernel_px, sample_names=sample_names, metric_names=metric_names,
                specimen_z_mm=column.specimen_z, defaults=SEMSettings().as_dict(), seed=seed, dose=dose)
    np.savez_compressed(out_dir / "sem_ray_db.npz", params=values.astype(np.float32), psfs=psfs,
                        images=images, samples=sample_stack.astype(np.float32), metrics=metrics,
                        meta=json.dumps(meta))
    with open(out_dir / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + names + metric_names)
        for i in range(n_configs):
            w.writerow([i] + [f"{v:.6g}" for v in values[i]] + [f"{v:.4g}" for v in metrics[i]])
    if verbose:
        print(f"saved {n_configs} configs to {out_dir.resolve()}")
    return out_dir / "sem_ray_db.npz"


def load_database(path="output/database/sem_ray_db.npz"):
    """Returns a dict with params, psfs, images, samples, metrics (arrays) and meta (dict)."""
    with np.load(path) as d:
        db = {k: d[k] for k in d.files if k != "meta"}
        db["meta"] = json.loads(str(d["meta"]))
    return db
