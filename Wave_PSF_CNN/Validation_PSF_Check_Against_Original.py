"""This file checks that Model_PSF_Wave_Optics gives exactly the same PSFs as the original
   calcPSFOTFwave in WavePSF0.py, for random aberrations, with and without source blur."""

#################################################################################################################################################################
"""Imports"""

# External imports
import numpy as np

# Internal imports
from constants import ORIGINAL_FILE, VOLTAGE, THETA
from Model_PSF_Wave_Optics import psf_field, electron_wavelength

#################################################################################################################################################################
"""Constants"""

N_TRIALS = 6                            # -     (random coefficient sets to compare)
N_GRID = 512                            # -     (grid size used for the comparison)
ZPF = 3                                 # -     (zero-pad factor used for the original)
TOLERANCE = 1e-3                        # -     (max allowed difference, relative to the PSF peak)
SEED = 0                                # -

# Typical size of every coefficient for the random trials (m), order = COEFF_NAMES
COEFF_SCALE = np.array([0.3e-6, 0.2e-6, 0.2e-6, 5e-6, 5e-6, 10e-6, 10e-6, 2e-3, 0.2e-3, 0.2e-3])

#################################################################################################################################################################
"""Functions"""

def load_original():
    """
    Load calcPSFOTFwave from WavePSF0.py without running its demo at the bottom (which opens a
    plot window). Only the part above the '# --- begin program ---' line is executed.

    Returns the function calcPSFOTFwave.
    """
    with open(ORIGINAL_FILE) as f:
        source = f.read().split('# -------------------------- begin program')[0]
    namespace = {}
    exec(source, namespace)
    return namespace['calcPSFOTFwave']

def compare(calcPSFOTFwave, coeffs, d50_geo):
    """
    Make the same PSF with the original and the new code and return the largest difference.

    calcPSFOTFwave : the original function
    coeffs         : the 10 coefficients (m)
    d50_geo        : source blur (m)

    Returns the max absolute difference, with both PSFs scaled to a peak of 1.
    """
    wavelength = float(electron_wavelength(VOLTAGE))
    original, _ = calcPSFOTFwave(N_GRID, ZPF, wavelength, THETA, d50_geo, coeffs)

    # Same pixel size as the original uses for this zero-pad factor
    pixel_size = 0.5 * wavelength / (ZPF * THETA)
    new = psf_field(coeffs, N_GRID, pixel_size, VOLTAGE, THETA, d50_geo)[0]
    new = new / new.max()
    return np.abs(new - original).max()

#################################################################################################################################################################
"""Main"""

def main():
    calcPSFOTFwave = load_original()
    rng = np.random.default_rng(SEED)

    worst = 0.0
    for trial in range(N_TRIALS):
        coeffs = rng.uniform(-1, 1, 10) * COEFF_SCALE
        for d50_geo in (0.0, 1e-9):
            difference = compare(calcPSFOTFwave, coeffs, d50_geo)
            worst = max(worst, difference)
            print(f'trial {trial}  d50_geo = {d50_geo:.0e} m   max |new - original| = {difference:.2e} of the peak')

    verdict = 'PASS' if worst < TOLERANCE else 'FAIL'
    print(f'{verdict}  (worst difference {worst:.2e}, tolerance {TOLERANCE:.0e})')

if __name__ == '__main__':
    main()
