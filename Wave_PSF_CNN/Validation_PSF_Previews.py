"""This file makes the figures used to check the PSF generation and the dataset (saved in Figures/)."""

#################################################################################################################################################################
"""Imports"""

# External imports
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')                   # only saving figures, no windows
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import torch

# Internal imports
from constants import (COEFF_NAMES, COEFF_RANGES, DEFOCUS_OFFSETS, PIXEL_SIZE, N_PIXELS, D50_GEO,
                       DATA_FOLDER, DATASET_NAME, FIGURES_FOLDER)
from Model_PSF_Wave_Optics import psf_stack
from Data_PSF_Generator import load_dataset, half_ranges
from Data_PSF_Torch_Dataset import PSFDataset

#################################################################################################################################################################
"""Info"""

# Figures:
#   01  the in-focus PSF for one coefficient at a time, at fractions of its range limit
#   02  the through-focus stack (the CNN input) for one coefficient at a time
#   03  why a stack is needed: flipping the sign of all even aberrations (C1, A1, C3, A3) leaves
#       a single in-focus PSF unchanged, but not the defocused ones
#   04  random samples read back through the PyTorch Dataset, clean and with shot noise
#       (exactly what the CNN will receive)
#   05  dataset statistics: are the coefficients evenly covered, and does every PSF fit the window
#
# Usage:
#   python Validation_PSF_Previews.py                 # uses Data/psf_v1 for figures 04-05
#   python Validation_PSF_Previews.py --name test

#################################################################################################################################################################
"""Plot styling"""

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 8,
    'figure.titlesize': 11,
})

PSF_COLOURMAP = 'inferno'
GAMMA = 0.5                             # -     (square-root colour scale: brightens the faint tails of the PSF)
DIFFERENCE_COLOURMAP = 'RdBu_r'
HISTOGRAM_COLOUR = 'tab:blue'
CONTAINED_COLOUR = 'tab:orange'
RADIUS_COLOUR = 'tab:green'
DPI = 120

HALF_WINDOW_NM = N_PIXELS * PIXEL_SIZE * 1e9 / 2        # nm  (half the stored window width)

# Units used to print each coefficient: (scale, unit)
PRINT_UNITS = {'C1': (1e-6, 'um'), 'A1': (1e-6, 'um'), 'B2': (1e-6, 'um'),
               'A2': (1e-6, 'um'), 'C3': (1e-3, 'mm'), 'A3': (1e-3, 'mm')}

#################################################################################################################################################################
"""Functions"""

def _show_psf(ax, image, title=None, gamma=GAMMA):
    """
    Show one PSF (or image) scaled to a peak of 1, with no axis ticks.

    ax    : matplotlib Axes
    image : 2D array
    title : optional title above the image
    gamma : colour scale exponent (1 = linear)
    """
    image = np.asarray(image, dtype=float)
    if image.max() > 0:
        image = image / image.max()
    ax.imshow(image, origin='lower', cmap=PSF_COLOURMAP, norm=PowerNorm(gamma, vmin=0, vmax=1),
              extent=(-HALF_WINDOW_NM, HALF_WINDOW_NM, -HALF_WINDOW_NM, HALF_WINDOW_NM),
              interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

def _coeff_label(name, value):
    """
    Short label for a coefficient, e.g. 'C1=+0.15 um'.

    name  : coefficient name (from COEFF_NAMES)
    value : coefficient (m)

    Returns the label string.
    """
    scale, unit = PRINT_UNITS[name[:2]]
    return f'{name}={value / scale:+.3g} {unit}'

def _plane_labels():
    """
    Returns a label per through-focus plane, e.g. ['C1 -0.2 um', 'in focus', 'C1 +0.2 um'].
    """
    labels = []
    for offset in DEFOCUS_OFFSETS:
        if offset == 0:
            labels.append('in focus')
        else:
            labels.append(f'C1 {offset * 1e6:+.1f} um')
    return labels

def _one_coefficient_at_a_time(fractions):
    """
    Coefficient sets where only one coefficient is non-zero, at fractions of its range limit.

    fractions : list of fractions, e.g. [-1, -0.5, 0, 0.5, 1]

    Returns array (10, len(fractions), 10): [which coefficient, which fraction, coefficients].
    """
    coeffs = np.zeros((10, len(fractions), 10))
    for i, name in enumerate(COEFF_NAMES):
        coeffs[i, :, i] = np.array(fractions) * COEFF_RANGES[name][1]
    return coeffs

def _save(fig, filename):
    """
    Save a figure in FIGURES_FOLDER and close it.
    """
    os.makedirs(FIGURES_FOLDER, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_FOLDER, filename), dpi=DPI)
    plt.close(fig)
    print(f'  wrote Figures/{filename}')

def figure_one_coefficient_in_focus():
    """
    Figure 01: in-focus PSF for every coefficient at -100%, -50%, 0, +50% and +100% of its range.
    """
    fractions = [-1, -0.5, 0, 0.5, 1]
    coeffs = _one_coefficient_at_a_time(fractions)
    images, _ = psf_stack(coeffs.reshape(-1, 10), through_focus=False)
    images = images.reshape(10, len(fractions), N_PIXELS, N_PIXELS)

    fig, axes = plt.subplots(10, len(fractions), figsize=(1.6 * len(fractions) + 0.6, 16.5))
    for i, name in enumerate(COEFF_NAMES):
        for j in range(len(fractions)):
            _show_psf(axes[i, j], images[i, j], _coeff_label(name, coeffs[i, j, i]))
    fig.suptitle(f'In-focus PSF, one coefficient at a time (others 0)\n'
                 f'window {2 * HALF_WINDOW_NM:.1f} nm, {N_PIXELS} px of {PIXEL_SIZE * 1e9:.2f} nm, '
                 f'd50_geo {D50_GEO * 1e9:g} nm, square-root colour scale')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, '01_one_coefficient_in_focus.png')

def figure_one_coefficient_through_focus():
    """
    Figure 02: the through-focus stack with one coefficient at 70% of its range limit.
    """
    coeffs = _one_coefficient_at_a_time([0.7])[:, 0]
    images, _ = psf_stack(coeffs)
    reference, _ = psf_stack(np.zeros((1, 10)))
    labels = _plane_labels()

    rows = [('no aberrations', reference[0])]
    for i, name in enumerate(COEFF_NAMES):
        rows.append((_coeff_label(name, coeffs[i, i]), images[i]))

    fig, axes = plt.subplots(len(rows), len(labels), figsize=(1.7 * len(labels) + 0.6, 1.65 * len(rows)))
    for r, (row_label, stack) in enumerate(rows):
        for j in range(len(labels)):
            _show_psf(axes[r, j], stack[j], labels[j] if r == 0 else None)
        axes[r, 0].set_ylabel(row_label, fontsize=7)
    fig.suptitle('Through-focus stack (the CNN input), one coefficient at 70% of its limit')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, '02_one_coefficient_through_focus.png')

def figure_sign_ambiguity():
    """
    Figure 03: a random coefficient set next to the same set with all even aberrations
    (C1, A1, C3, A3) sign-flipped. The in-focus PSFs are identical, the defocused ones are not.
    """
    rng = np.random.default_rng(3)
    coeffs = np.array([0.6 * rng.uniform(*COEFF_RANGES[name]) for name in COEFF_NAMES])
    flipped = coeffs.copy()
    for i, name in enumerate(COEFF_NAMES):
        if name[:2] in ('C1', 'A1', 'C3', 'A3'):
            flipped[i] = -coeffs[i]
    images, _ = psf_stack(np.stack([coeffs, flipped]))
    labels = _plane_labels()

    fig, axes = plt.subplots(3, len(labels), figsize=(2.3 * len(labels), 7.2))
    limit = np.abs(images[0] - images[1]).max()         # one colour scale for all difference panels
    for j in range(len(labels)):
        _show_psf(axes[0, j], images[0, j], labels[j])
        _show_psf(axes[1, j], images[1, j])
        difference = images[0, j] - images[1, j]
        axes[2, j].imshow(difference, origin='lower', cmap=DIFFERENCE_COLOURMAP, vmin=-limit, vmax=limit)
        axes[2, j].set_xticks([])
        axes[2, j].set_yticks([])
        axes[2, j].set_title(f'difference: max {np.abs(difference).max() / images[0, j].max():.1e} of peak')
    axes[0, 0].set_ylabel('coefficients c')
    axes[1, 0].set_ylabel('c with C1, A1, C3, A3\nsign-flipped')
    axes[2, 0].set_ylabel('difference')
    fig.suptitle('A single in-focus PSF cannot tell the sign of the even aberrations (identical middle column);\n'
                 'the defocused planes can, which is why the CNN gets a through-focus stack')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, '03_sign_ambiguity.png')

def figure_dataset_samples(folder, n_rows=8, dose=200):
    """
    Figure 04: random samples read through the PyTorch Dataset, clean (left) and with Poisson
    shot noise at the given dose (right), with their coefficients.

    folder : dataset folder
    n_rows : number of samples shown
    dose   : counts at the brightest pixel for the noisy version
    """
    clean = PSFDataset(folder)
    noisy = PSFDataset(folder, dose=dose)
    torch.manual_seed(0)
    ids = np.random.default_rng(1).choice(len(clean), n_rows, replace=False)
    labels = _plane_labels()
    P = len(labels)

    fig, axes = plt.subplots(n_rows, 2 * P + 1, figsize=(1.55 * 2 * P + 3.2, 1.6 * n_rows),
                             gridspec_kw={'width_ratios': [1] * (2 * P) + [2.1]})
    for r, i in enumerate(ids):
        x, y = clean[i]
        x_noisy, _ = noisy[i]
        for j in range(P):
            _show_psf(axes[r, j], x[j], labels[j] if r == 0 else None)
            _show_psf(axes[r, P + j], x_noisy[j], f'{labels[j]}\nPoisson, peak {dose}' if r == 0 else None, gamma=1)
        coeffs = clean.to_coeffs(y.numpy())
        text = '\n'.join(_coeff_label(name, value) for name, value in zip(COEFF_NAMES, coeffs))
        axes[r, -1].axis('off')
        axes[r, -1].text(0, 0.5, text, fontsize=5.5, family='monospace', va='center')
        axes[r, 0].set_ylabel(f'sample {i}', fontsize=7)
    fig.suptitle('Random samples read back through PSFDataset (left: clean, square-root scale; '
                 'right: with shot noise, linear)')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, '04_dataset_samples.png')

def figure_dataset_statistics(folder, n_radius=500):
    """
    Figure 05: histograms of every coefficient (should be flat), of the fraction of the beam
    current inside the window, and of the rms radius of the in-focus PSF.

    folder   : dataset folder
    n_radius : number of samples used for the radius histogram
    """
    dataset = load_dataset(folder)
    scaled_coeffs = dataset['coeffs'] / half_ranges(dataset['meta'])
    in_focus = DEFOCUS_OFFSETS.index(0.0)

    # rms radius of the in-focus PSF around its centre of mass
    yy, xx = np.indices((N_PIXELS, N_PIXELS))
    radii_nm = []
    for i in np.linspace(0, len(scaled_coeffs) - 1, min(n_radius, len(scaled_coeffs))).astype(int):
        image = dataset['images'][i, in_focus].astype(float)
        total = image.sum()
        x_centre = (image * xx).sum() / total
        y_centre = (image * yy).sum() / total
        r2 = (image * ((xx - x_centre) ** 2 + (yy - y_centre) ** 2)).sum() / total
        radii_nm.append(np.sqrt(r2) * PIXEL_SIZE * 1e9)

    fig, axes = plt.subplots(2, 6, figsize=(16, 5.5))
    for i, name in enumerate(COEFF_NAMES):
        ax = axes.flat[i]
        ax.hist(scaled_coeffs[:, i], bins=30, color=HISTOGRAM_COLOUR, alpha=0.8)
        ax.set_title(f'{name} / half its range')
    axes.flat[10].hist(dataset['contained'].min(axis=1), bins=30, color=CONTAINED_COLOUR)
    axes.flat[10].set_title('fraction of the beam inside the window\n(worst plane per sample)')
    axes.flat[11].hist(radii_nm, bins=30, color=RADIUS_COLOUR)
    axes.flat[11].set_title(f'rms radius of the in-focus PSF (nm)\n({len(radii_nm)} samples)')
    for ax in axes.flat:
        ax.tick_params(labelsize=6)
    fig.suptitle(f'Dataset statistics ({len(scaled_coeffs)} samples, '
                 f'{100 * dataset["meta"]["redraw_fraction"]:.1f}% of draws redrawn because the PSF did not fit the window)')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, '05_dataset_statistics.png')

#################################################################################################################################################################
"""Main"""

def main():
    p = argparse.ArgumentParser(description='Figures to check the PSF generation and the dataset')
    p.add_argument('--name', type=str, default=DATASET_NAME, help=f'dataset folder in Data/ (default {DATASET_NAME})')
    args = p.parse_args()

    # Step 1: figures straight from the PSF model
    figure_one_coefficient_in_focus()
    figure_one_coefficient_through_focus()
    figure_sign_ambiguity()

    # Step 2: figures of the stored dataset (only if it has been generated)
    folder = os.path.join(DATA_FOLDER, args.name)
    if os.path.exists(os.path.join(folder, 'meta.json')):
        figure_dataset_samples(folder)
        figure_dataset_statistics(folder)
    else:
        print(f'  (no dataset in Data/{args.name}: run Data_PSF_Generator.py first for figures 04-05)')

if __name__ == '__main__':
    main()
