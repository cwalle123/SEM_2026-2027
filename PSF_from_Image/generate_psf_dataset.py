# External imports
import json
import os
import time
import multiprocessing as mp
import warnings
import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
from pathlib import Path

##################################################################################################################################################################################################################
"""Constants"""

electron_charge = 1.602176634e-19       # [C]
V = 30000                               # Voltage [V]
h = 6.62607015e-34                      # Planck's constant [J*s]
c = 299792458                           # Speed of light [m/s]
electron_mass = 9.10938356e-31          # Electron mass [kg]

# Fixed defocus series: every aberration vector is imaged at these three C1 values
defocus_values = np.array([5e-6, 0.0, -5e-6])       # C1 [m]: +5 um, 0, -5 um

# Full aberration vector order: [C1, A1r, A1i, B2r, B2i, A2r, A2i, C3, A3r, A3i]   (C3 comes before A3)
vector_labels = ["C1", "A1r", "A1i", "B2r", "B2i", "A2r", "A2i", "C3", "A3r", "A3i"]

# Random coefficients shared by the 3 vectors of a set (everything except C1, which is set by defocus_values)
aberration_labels = vector_labels[1:]
# Each coefficient is sampled uniformly in [low, high] [m]. Values for an uncorrected SEM objective lens (30 kV, alpha = 20 mrad).
# For reference, the pi/4 tolerances at 30 kV / 20 mrad are roughly: A1 ~4 nm, B2 and A2 ~0.3 um, C3 and A3 ~20 um,
# so all ranges below are clearly visible in the PSF.
# Sources:
#   [1] Uno, Honda, Nakamura, Matsuya & Zach, "Aberration correction and its automatic control in scanning electron microscopes",
#       Optik 116 (2005) 438-448. Through-focus SEM aberration retrieval for C1, A1, B2, A2, Cs; typical orders of magnitude
#       A1 ~1e-8, B2 ~1e-7, A2 ~1e-6, Cs ~1e-3 m (as tabulated by Dortland, TU Delft BEP "A benchmark study on SEM aberration retrieval", 2024,
#       who measured A1 ~93 nm, B2 ~0.1 um, A2 ~2 um, Cs ~1 mm on a real SEM and tested B2 up to 8 um, A2 up to 20 um, Cs up to 5 mm).
#   [2] Kazumori et al., Microsc. Microanal. 6 (S2) (2000) 764: JSM-6700F conical objective, Cs = 1.9 mm at WD = 3 mm.
#       JEOL SEM glossary: out-lens objective at WD ~10 mm has Cs ~50 mm (only usable with alpha ~5 mrad, not 20 mrad).
#       Cs scales roughly with WD^3 (Microscopy Today, "Some things you might like to know about electron lenses").
#   [3] Rudnaya, "Automated focusing and astigmatism correction in electron microscopy", PhD thesis TU/e (2011):
#       defocus and two-fold astigmatism drift and must be re-corrected regularly, so A1 can be large before stigmation.
#   [4] Krivanek, Dellby & Lupini, "Towards sub-A electron beams", Ultramicroscopy 78 (1999) 1: without correction,
#       aberrations other than C1 and C3 are usually much smaller and are left out; A3 is a third-order parasitic of the objective.
# Crop budget: the crop is +-172.8 nm and the 5 um defocus alone already spreads the beam over +-100 nm (C1 * alpha),
# so the summed worst-case ray deviations of all other ranges are kept below ~60 nm (checked numerically: >99% energy in the crop).
#                                low      high
aberration_bounds = np.array([[  -1e-6,    1e-6],       # A1r   2-fold astigmatism: ~10-100 nm after stigmation [1], up to ~um before/with drift [3]
                              [  -1e-6,    1e-6],       # A1i
                              [ -10e-6,   10e-6],       # B2r   axial coma: ~0.1 um when aligned [1]; misalignment tilt tau gives B2 ~ Cs*tau ~ 2 mm * 5 mrad = 10 um
                              [ -10e-6,   10e-6],       # B2i
                              [ -20e-6,   20e-6],       # A2r   3-fold astigmatism: ~1-2 um measured [1]; 20 um = upper end tested in [1]-based benchmark
                              [ -20e-6,   20e-6],       # A2i
                              [ 0.5e-3,    5e-3],       # C3    spherical: 0.5 mm (well-designed pole piece) to 5 mm (longer WD) [2]; always > 0 (Scherzer)
                              [-200e-6,  200e-6],       # A3r   4-fold astigmatism: parasitic, rarely measured on SEMs [4]; ~10x the pi/4 tolerance
                              [-200e-6,  200e-6]])      # A3i

# Which coefficients are random in each dataset type. Inactive coefficients are set to exactly 0.
#   3: A1 + C3      -> defocus, 2-fold astigmatism and spherical aberration: the classic uncorrected-SEM probe model; C3 is
#                      unavoidable for round lenses and A1 is the aberration the operator corrects every session [3], [4]
#   7: + B2 + A2    -> exactly the 5 aberrations (C1, A1, B2, A2, Cs) measured from SEM through-focus series by Uno et al. [1]:
#                      all axial aberrations up to second order plus the third-order Cs
#   9: + A3         -> everything up to third order in this model
coefficient_sets = {3: ["A1r", "A1i", "C3"],
                    7: ["A1r", "A1i", "B2r", "B2i", "A2r", "A2i", "C3"],
                    9: aberration_labels}

def dataset_name(n_coefficients, n_samples):
    """Clear filename, e.g. 'psf_3coef_A1-C3_n1000.npz' or 'psf_7coef_A1-B2-A2-C3_n1000.npz'."""
    groups = list(dict.fromkeys(label.rstrip("ri") for label in coefficient_sets[n_coefficients]))   # A1r, A1i -> A1 (C3 stays C3)
    return f"psf_{n_coefficients}coef_{'-'.join(groups)}_n{n_samples}.npz"

##################################################################################################################################################################################################################
"""PSF generation"""

def electron_wavelength(voltage):
    """Relativistic electron wavelength [m] for an accelerating voltage [V]."""
    return h / np.sqrt(2 * electron_mass * electron_charge * voltage + (1 / c**2) * (electron_charge * voltage)**2)

def aberration_function(tx, ty, alpha, aberration_vector):
    """phi(x', y') [m]: Zernike-balanced wave aberration over a circular aperture.
    tx, ty = ray angles [rad], alpha = half opening angle [rad].
    aberration_vector = [C1, A1r, A1i, B2r, B2i, A2r, A2i, C3, A3r, A3i] in meters."""
    C1, A1r, A1i, B2r, B2i, A2r, A2i, C3, A3r, A3i = aberration_vector

    tx2 = tx**2;   ty2 = ty**2;   al2 = alpha**2
    S  = 0.5*C1*(tx2 + ty2) + 0.5*A1r*(tx2 - ty2) + A1i*tx*ty                    # defocus, 2-fold astigmatism:   Re{1/2 w w* C1 + 1/2 w*^2 A1}
    S += (B2r*tx - B2i*ty)*(tx2 + ty2 - (2/3)*al2)                               # coma (tilt-balanced):          Re{w^2 w* B2}
    S += A2r*tx*((tx2/3) - ty2) + A2i*ty*(tx2 - (ty2/3))                         # 3-fold astigmatism:            Re{1/3 w*^3 A2}
    S += 0.25*C3*(tx2 + ty2)*(tx2 + ty2 - al2)                                   # spherical (defocus-balanced):  Re{1/4 (w w*)^2 C3}
    S += 0.25*(A3r*(tx**4 + ty**4 - 6*tx2*ty2) + A3i*(4*tx*ty*(tx2 - ty2)))      # 4-fold astigmatism:            Re{1/4 w*^4 A3}
    return S

class PSFGenerator:
    """Makes intensity PSFs for one fixed microscope setting (voltage, alpha, pixel_size, n, crop).
    Everything that is the same for every PSF is computed once here, so each call only does the work that changes:
      - the aberration polynomial is split into 10 fixed basis maps (one per coefficient), so the phase is one matrix product
      - the phase is only evaluated in the small square around the aperture, not on the whole n x n grid
      - phase, pupil and FFT are single precision (float32 / complex64), plenty for a PSF that is binned afterwards
      - only the crop window is turned into intensity; the normalization comes from Parseval, not from summing the full PSF."""

    def __init__(self, voltage, alpha, pixel_size, n, crop, fft_workers=-1):
        wavelength = electron_wavelength(voltage)
        dtheta = wavelength / (n * pixel_size)          # angular step, chosen so that the FFT lands on image pixels of size pixel_size
        if n * dtheta / 2 < alpha:
            raise ValueError(f"Aperture doesn't fit: need pixel_size < {wavelength / (2 * alpha):.3e} m")

        # Square window around the aperture inside the full n x n pupil grid
        half = int(np.ceil(alpha / dtheta)) + 1
        self.box = slice(n // 2 - half, n // 2 + half + 1)
        theta = (np.arange(-half, half + 1)) * dtheta
        TX, TY = np.meshgrid(theta, theta)
        self.box_shape = TX.shape
        self.aperture = np.hypot(TX, TY) <= alpha
        self.aperture_f32 = self.aperture.astype(np.float32)

        # Basis map k = phase [rad] for a coefficient vector with 1 m in position k and 0 elsewhere (the polynomial is linear in the coefficients)
        basis = np.stack([2 * np.pi / wavelength * aberration_function(TX, TY, alpha, np.eye(10)[k]) for k in range(10)], axis=-1)
        self.basis = basis.reshape(-1, 10).astype(np.float32)

        self.n = n
        self.fft_workers = fft_workers
        self.crop = slice(n // 2 - crop, n // 2 + crop)
        self.pupil = np.zeros((n, n), dtype=np.complex64)               # reused buffer, zero outside the aperture
        self.norm = n**2 * self.aperture.sum()                          # Parseval: sum |FT{P}|^2 = n^2 * sum |P|^2

    def __call__(self, aberration_vector):
        """Aberration vector [m] -> intensity PSF in the crop window (2*crop x 2*crop), normalized so the full PSF sums to 1."""
        phase = (self.basis @ np.asarray(aberration_vector, dtype=np.float32)).reshape(self.box_shape)

        # Sampling check on a 4x coarser grid (the phase is a smooth polynomial, so its gradient changes slowly): 16x cheaper
        k = 4
        gy, gx = np.gradient(phase[::k, ::k])
        max_step = np.max(np.hypot(gx, gy)[self.aperture[::k, ::k]]) / k
        if max_step > np.pi:
            warnings.warn(f"Pupil phase undersampled ({max_step:.1f} rad/px > pi): increase n * pixel_size")

        box = self.pupil[self.box, self.box]                                     # view into the pupil buffer
        box.real = np.cos(phase) * self.aperture_f32                             # pupil function P(x', y') = A exp(j phase)
        box.imag = np.sin(phase) * self.aperture_f32
        U = scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.ifftshift(self.pupil), workers=self.fft_workers))   # complex PSF U(x, y) = FT{P}
        U_crop = U[self.crop, self.crop]
        return (U_crop.real**2 + U_crop.imag**2).astype(np.float64) / self.norm                  # intensity PSF in the crop window

def bin_image(image, out_size):
    """Downsamples a square image to (out_size, out_size) by summing blocks of pixels (keeps the total intensity)."""
    factor = image.shape[0] // out_size
    return image[:out_size * factor, :out_size * factor].reshape(out_size, factor, out_size, factor).sum(axis=(1, 3))

##################################################################################################################################################################################################################
"""Dataset generation"""

def make_aberration_sets(n_samples, active=aberration_labels, defocus=defocus_values, bounds=aberration_bounds, seed=0):
    """Random aberration sets. Each set = len(defocus) full vectors with the SAME random coefficients, only C1 differs.
    Only the coefficients in `active` are random; all others are exactly 0.
    Returns shared (n_samples, 9) and sets (n_samples, len(defocus), 10)."""
    rng = np.random.default_rng(seed)
    mask = np.isin(aberration_labels, active)
    shared = np.zeros((n_samples, len(bounds)))
    shared[:, mask] = rng.uniform(bounds[mask, 0], bounds[mask, 1], size=(n_samples, mask.sum()))
    sets = np.empty((n_samples, len(defocus), len(bounds) + 1))
    sets[:, :, 0] = defocus                     # C1 per vector in the set
    sets[:, :, 1:] = shared[:, None, :]         # same other coefficients for all vectors in the set
    return shared, sets

_worker_generator = None

def _init_worker(voltage, alpha, pixel_size, n, crop):
    """Runs once in every worker process: builds that worker's own PSFGenerator (1 FFT thread each, the processes give the parallelism)."""
    global _worker_generator
    _worker_generator = PSFGenerator(voltage, alpha, pixel_size, n, crop, fft_workers=1)

def _make_set(args):
    """Worker task: all PSFs of one aberration set -> (index, binned images, fraction of energy in the crop per channel)."""
    i, aberration_set, out_size = args
    psis = [_worker_generator(vector) for vector in aberration_set]
    return i, np.stack([bin_image(p, out_size) for p in psis]).astype(np.float32), [p.sum() for p in psis]

def format_time(seconds):
    """Seconds -> 'h:mm:ss'."""
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"

def generate_training_dataset(n_samples, n_coefficients=9, out_size=128, crop=1152, voltage=V, alpha=20e-3, pixel_size=150e-12, n=3072,
                              defocus=defocus_values, bounds=aberration_bounds, val_fraction=0.1, test_fraction=0.1, seed=0,
                              filename=None, folder="Data", n_preview=3, checkpoint_every=10, n_workers=None):
    """Generates n_samples sets of 3 aberration vectors (same coefficients, C1 = +5 um, 0, -5 um) and saves everything in one file.
    n_coefficients = 3, 7 or 9: which coefficients are random (see coefficient_sets); all others are exactly 0.
    filename = None gives a clear automatic name, e.g. psf_3coef_A1-C3_n1000.npz. The file contains:
      images              (N, 3, out_size, out_size) float32  -> one PSF per vector in the set, binned from the (2*crop)^2 center
      aberration_vectors  (N, 3, 10)                          -> the full vectors [C1, A1r, ..., A3i] of each set [m] (inactive ones = 0)
      labels              (N, K) float32                      -> the K active coefficients scaled to [-1, 1] within their bounds, what the network predicts
      defocus_values      (3,)                                -> C1 of each channel [m]
      bounds, label_names                                     -> (K, 2) and (K,) for the active coefficients, to convert labels back to meters:
                                                                 low + (labels + 1) / 2 * (high - low)
      n_coefficients, active_mask                             -> which of the 9 coefficients were random (mask over [A1r, ..., A3i])
      train_idx, val_idx, test_idx                            -> fixed split, so every training run uses the same test set
      voltage, alpha, pixel_size, binned_pixel_size           -> microscope settings the images were made with
    The first n_preview stacks are also saved as PNG for a quick look (in Data/<dataset name>_previews/).
    Progress is cached in the Data folder every checkpoint_every sets. If the run stops, just run it again with the same
    settings and it continues where it stopped. The cache files are deleted once the final .npz is saved.
    n_workers = number of parallel processes (default: half the CPU threads). Each one uses roughly 0.5-1 GB of RAM,
    so lower it if your computer runs out of memory.
    """
    folder = Path(__file__).parent / folder
    folder.mkdir(exist_ok=True)
    if (2 * crop) % out_size != 0 or crop > n // 2:
        raise ValueError("2*crop must be divisible by out_size and crop must be <= n // 2")

    if n_coefficients not in coefficient_sets:
        raise ValueError(f"n_coefficients must be one of {list(coefficient_sets)}")
    active = coefficient_sets[n_coefficients]
    active_mask = np.isin(aberration_labels, active)
    if filename is None:
        filename = dataset_name(n_coefficients, n_samples)
    print(f"Dataset {filename}: random coefficients {active}, all others 0")

    shared, aberration_vectors = make_aberration_sets(n_samples, active, defocus, bounds, seed)   # deterministic from seed, so a resumed run gets the same vectors

    # ----- Cache: images are written straight to a .npy file on disk, progress is saved every checkpoint_every sets -----
    stem = Path(filename).stem
    cache_images = folder / f"{stem}_partial_images.npy"
    cache_progress = folder / f"{stem}_partial_progress.json"
    preview_folder = folder / f"{stem}_previews"
    if n_preview > 0:
        preview_folder.mkdir(exist_ok=True)
    settings = dict(n_samples=n_samples, n_coefficients=n_coefficients, out_size=out_size, crop=crop, voltage=voltage, alpha=alpha,
                    pixel_size=pixel_size, n=n, defocus=np.asarray(defocus).tolist(), bounds=np.asarray(bounds).tolist(), seed=seed)
    shape = (n_samples, len(defocus), out_size, out_size)

    n_done = 0
    if cache_progress.exists() and cache_images.exists():
        progress = json.loads(cache_progress.read_text())
        if progress["settings"] != settings:
            raise ValueError(f"Found a cache made with different settings: delete {cache_images.name} and {cache_progress.name}, "
                             f"or use the same settings / another filename")
        images = np.lib.format.open_memmap(cache_images, mode="r+")
        n_done = progress["n_done"]
        print(f"Resuming from cache: {n_done}/{n_samples} PSF sets already done")
    else:
        images = np.lib.format.open_memmap(cache_images, mode="w+", dtype=np.float32, shape=shape)

    def save_progress(done):
        images.flush()                                                     # images first, so the counter never points past saved data
        tmp = cache_progress.with_suffix(".tmp")
        tmp.write_text(json.dumps({"n_done": done, "settings": settings}))
        tmp.replace(cache_progress)                                        # atomic, so a crash while writing can't corrupt the progress file

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) // 2)
    print(f"Using {n_workers} worker process(es); setting up the PSF generator takes a few seconds per worker")

    tasks = ((i, aberration_vectors[i], out_size) for i in range(n_done, n_samples))
    start = None
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(voltage, alpha, pixel_size, n, crop)) as pool:
        for i, stack, energy in pool.imap(_make_set, tasks):              # imap keeps the order, so the cache always holds sets 0..done-1
            if start is None:
                start, first = time.perf_counter(), i                     # time from the first finished set, so the setup doesn't skew the estimate
            for j, e in enumerate(energy):
                if e < 0.99:
                    warnings.warn(f"Sample {i}, channel {j}: {100 * (1 - e):.1f}% of the PSF falls outside the crop window, increase crop (and n)")
            images[i] = stack
            if i < n_preview:
                plt.imsave(preview_folder / f"preview_{i:05d}.png", np.hstack([im / im.max() for im in stack]), cmap="inferno", origin="lower")   # each panel scaled to its own max (preview only)

            done = i + 1
            if done % checkpoint_every == 0 or done == n_samples:
                save_progress(done)
            if done % 10 == 0 or done == n_samples:
                elapsed = time.perf_counter() - start
                rate = elapsed / max(done - first - 1, 1)                 # seconds per set in this run
                print(f"\rGenerated {done}/{n_samples} PSF sets ({100 * done / n_samples:.1f}%) | "
                      f"elapsed {format_time(elapsed)} | remaining ~{format_time(rate * (n_samples - done))}", end="", flush=True)
    print()

    active_bounds = bounds[active_mask]
    low, high = active_bounds[:, 0], active_bounds[:, 1]
    labels = (2 * (shared[:, active_mask] - low) / (high - low) - 1).astype(np.float32)   # only the active coefficients, all on the same [-1, 1] scale,
                                                                                            # otherwise C3 (e-3) dominates the loss over A1 (e-7)

    idx = np.random.default_rng(seed + 1).permutation(n_samples)
    n_test = int(test_fraction * n_samples)
    n_val = int(val_fraction * n_samples)
    test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]

    np.savez_compressed(folder / filename, images=np.asarray(images), labels=labels, aberration_vectors=aberration_vectors,
                        defocus_values=np.asarray(defocus), bounds=active_bounds, label_names=np.array(active), vector_labels=np.array(vector_labels),
                        n_coefficients=n_coefficients, active_mask=active_mask,
                        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                        voltage=voltage, alpha=alpha, pixel_size=pixel_size,
                        binned_pixel_size=pixel_size * (2 * crop // out_size))
    print(f"Saved {n_samples} samples to {folder / filename}")

    del images                                                             # close the memmap before deleting the cache
    cache_images.unlink(); cache_progress.unlink()

#########################################################################################################
"""Run this file"""

def main():
    # n_coefficients = 3 (A1, C3), 7 (A1, B2, A2, C3) or 9 (all). Every type gets its own file, cache and preview folder,
    # e.g. Data/psf_3coef_A1-C3_n1000.npz, Data/psf_7coef_A1-B2-A2-C3_n1000.npz, Data/psf_9coef_A1-B2-A2-C3-A3_n1000.npz
    generate_training_dataset(n_samples=1000, n_coefficients=9)

if __name__ == "__main__":
    main()
