"""This file generates the PSF dataset for the CNN (random aberrations -> through-focus PSF stacks)
   and loads it back."""

#################################################################################################################################################################
"""Imports"""

# External imports
import argparse
import json
import os
import time
import numpy as np

# Internal imports
from constants import (COEFF_NAMES, COEFF_RANGES, MACHINE_RANGES, DEFOCUS_OFFSETS, N_PIXELS, PIXEL_SIZE,
                       N_SAMPLES, DATASET_NAME, BATCH_SIZE, MIN_CONTAINED, SEED, DATA_FOLDER,
                       VOLTAGE, THETA, D50_GEO, BIN_FACTOR, PAD_FACTOR)
from Model_PSF_Wave_Optics import psf_stack

#################################################################################################################################################################
"""Info"""

# Workflow (main):
#   1. Draw random aberration coefficients (uniform within COEFF_RANGES) and machine settings
#      (within MACHINE_RANGES) for a batch of samples.
#   2. Make the through-focus PSF stack of every sample (Model_PSF_Wave_Optics.psf_stack).
#   3. Throw away samples whose PSF does not fit in the stored window (less than MIN_CONTAINED of
#      the beam current inside, in any plane) and draw new ones, so every stored PSF is complete.
#   4. Save, and repeat until there are enough samples.
#
# Usage:
#   python Data_PSF_Generator.py                      # N_SAMPLES samples -> Data/psf_v1
#   python Data_PSF_Generator.py --n 500 --name test  # small test set -> Data/test
#
# What is saved in Data/<name>/ (plain .npy files, read them with load_dataset below):
#   images.npy     (N, planes, 128, 128)  float16   every image scaled to a peak of 1
#   scales.npy     (N, planes)            float32   the peak value of every image, so
#                                                   images * scales = fraction of the beam current per pixel
#   coeffs.npy     (N, 10)                float64   aberration coefficients (m), order = COEFF_NAMES
#   machine.npy    (N, 3)                 float64   voltage (V), theta (rad), d50_geo (m) per sample
#   contained.npy  (N, planes)            float32   fraction of the beam current inside the window
#   meta.json                                       all settings used, so the dataset explains itself
#
# The images are stored as float16 (half the disk space); scaling every image to a peak of 1
# first keeps them accurate, since float16 loses precision for very small numbers.

#################################################################################################################################################################
"""Functions"""

def sample_uniform(rng, ranges, names, n):
    """
    Draw n random values for every quantity in names, each uniformly within its range.

    rng    : numpy random generator
    ranges : dict name -> (low, high)
    names  : the names to draw, in this order
    n      : number of samples

    Returns array (n, len(names)).
    """
    low = np.array([ranges[name][0] for name in names])
    high = np.array([ranges[name][1] for name in names])
    return low + rng.random((n, len(names))) * (high - low)

def generate_dataset(folder, n_samples, seed=SEED):
    """
    Generate a dataset and save it in folder (see the Info section for the files).

    folder    : folder to save in (made if it does not exist)
    n_samples : number of samples to store
    seed      : random seed

    Returns the fraction of drawn samples that had to be redrawn because they did not fit.
    """
    os.makedirs(folder, exist_ok=True)
    n_planes = len(DEFOCUS_OFFSETS)
    machine_names = list(MACHINE_RANGES)

    # The arrays are written straight to disk (memory-mapped), so a big dataset never has to fit in memory
    images = np.lib.format.open_memmap(os.path.join(folder, 'images.npy'), mode='w+',
                                       dtype=np.float16, shape=(n_samples, n_planes, N_PIXELS, N_PIXELS))
    scales = np.zeros((n_samples, n_planes), dtype=np.float32)
    coeffs = np.zeros((n_samples, 10))
    machine = np.zeros((n_samples, 3))
    contained = np.zeros((n_samples, n_planes), dtype=np.float32)

    rng = np.random.default_rng(seed)
    n_filled = 0                    # samples stored so far
    n_drawn = 0                     # samples drawn so far (including the redrawn ones)
    t_start = time.perf_counter()

    while n_filled < n_samples:
        # Step 1: random coefficients and machine settings for one batch
        batch_coeffs = sample_uniform(rng, COEFF_RANGES, COEFF_NAMES, BATCH_SIZE)
        batch_machine = sample_uniform(rng, MACHINE_RANGES, machine_names, BATCH_SIZE)

        # Step 2: through-focus PSF stacks
        batch_images, batch_contained = psf_stack(batch_coeffs, voltage=batch_machine[:, 0],
                                                  theta=batch_machine[:, 1], d50_geo=batch_machine[:, 2])
        n_drawn += BATCH_SIZE

        # Step 3: keep only the samples that fit in the window (worst plane counts)
        keep = np.flatnonzero(batch_contained.min(axis=1) >= MIN_CONTAINED)
        keep = keep[:n_samples - n_filled]                      # don't overshoot on the last batch

        # Step 4: store them
        stored = slice(n_filled, n_filled + len(keep))
        peak = batch_images[keep].max(axis=(-2, -1))
        images[stored] = (batch_images[keep] / peak[:, :, None, None]).astype(np.float16)
        scales[stored] = peak
        coeffs[stored] = batch_coeffs[keep]
        machine[stored] = batch_machine[keep]
        contained[stored] = batch_contained[keep]
        n_filled += len(keep)

        elapsed = time.perf_counter() - t_start
        time_left = elapsed / n_filled * (n_samples - n_filled) if n_filled else 0
        print(f'\r  {n_filled}/{n_samples} samples  ({elapsed:.0f} s, ~{time_left:.0f} s left, '
              f'{100 * (1 - n_filled / n_drawn):.1f}% redrawn)', end='', flush=True)
    print()

    images.flush()
    np.save(os.path.join(folder, 'scales.npy'), scales)
    np.save(os.path.join(folder, 'coeffs.npy'), coeffs)
    np.save(os.path.join(folder, 'machine.npy'), machine)
    np.save(os.path.join(folder, 'contained.npy'), contained)

    redraw_fraction = 1 - n_samples / n_drawn
    meta = {
        'n_samples': n_samples,
        'coeff_names': COEFF_NAMES,
        'coeff_units': 'm',
        'coeff_ranges': COEFF_RANGES,
        'machine_names': machine_names,
        'machine_units': ['V', 'rad', 'm'],
        'machine_ranges': MACHINE_RANGES,
        'nominal_machine': {'voltage': VOLTAGE, 'theta': THETA, 'd50_geo': D50_GEO},
        'defocus_offsets_m': DEFOCUS_OFFSETS,
        'pixel_size_m': PIXEL_SIZE,
        'n_pixels': N_PIXELS,
        'bin_factor': BIN_FACTOR,
        'pad_factor': PAD_FACTOR,
        'min_contained': MIN_CONTAINED,
        'seed': seed,
        'redraw_fraction': redraw_fraction,
    }
    with open(os.path.join(folder, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return redraw_fraction

def load_dataset(folder):
    """
    Load a generated dataset. The images are memory-mapped: they stay on disk and are only read
    when used, so a 1 GB dataset can be opened instantly.

    folder : dataset folder, e.g. Data/psf_v1

    Returns a dict with the arrays 'images', 'scales', 'coeffs', 'machine', 'contained' and the
    settings dict 'meta' (see the Info section).
    """
    dataset = {'images': np.load(os.path.join(folder, 'images.npy'), mmap_mode='r')}
    for name in ('scales', 'coeffs', 'machine', 'contained'):
        dataset[name] = np.load(os.path.join(folder, f'{name}.npy'))
    with open(os.path.join(folder, 'meta.json')) as f:
        dataset['meta'] = json.load(f)
    return dataset

def half_ranges(meta):
    """
    Half the width of every coefficient's range. Dividing the coefficients by this puts them
    roughly between -1 and 1, which is the scale the CNN learns on.

    meta : the 'meta' dict of a dataset

    Returns array (10,) in m.
    """
    ranges = meta['coeff_ranges']
    return np.array([(ranges[name][1] - ranges[name][0]) / 2 for name in meta['coeff_names']])

#################################################################################################################################################################
"""Main"""

def main():
    """
    Generate a dataset with the settings in constants.py.
    """
    p = argparse.ArgumentParser(description='Generate a through-focus PSF dataset for the CNN')
    p.add_argument('--n', type=int, default=N_SAMPLES, help=f'number of samples (default {N_SAMPLES})')
    p.add_argument('--name', type=str, default=DATASET_NAME, help=f'folder name in Data/ (default {DATASET_NAME})')
    p.add_argument('--seed', type=int, default=SEED, help=f'random seed (default {SEED})')
    args = p.parse_args()

    folder = os.path.join(DATA_FOLDER, args.name)
    print(f'Generating {args.n} samples -> {folder}')
    redraw_fraction = generate_dataset(folder, args.n, args.seed)
    print(f'Done. {100 * redraw_fraction:.1f}% of the drawn samples did not fit in the window and were redrawn.')

if __name__ == '__main__':
    main()
