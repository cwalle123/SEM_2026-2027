"""This file holds every setting of the PSF generation in one place: the SEM machine, how the
   images are sampled, the through-focus planes, the aberration ranges and the file locations."""

#################################################################################################################################################################
"""Imports"""

# External imports
import os

#################################################################################################################################################################
"""Physical constants"""

# Relativistic electron wavelength (same formula as WavePSF0.py):
#     lambda = 1.226e-9 / sqrt(U + EPS_RELATIVISTIC * U^2)      with U in V and lambda in m
WAVELENGTH_CONSTANT = 1.226e-9          # m V^0.5   (h / sqrt(2 m_e e), the non-relativistic part)
EPS_RELATIVISTIC = 9.785e-7             # 1/V       (e / (2 m_e c^2), the relativistic correction)

#################################################################################################################################################################
"""SEM machine"""

# Nominal machine settings. Every sample in the dataset stores its own values, so these can
# later be turned into ranges (see MACHINE_RANGES) to match real images taken at other settings.
VOLTAGE = 30000.0                       # V     (acceleration voltage)
THETA = 0.015                           # rad   (half opening angle of the beam, set by the aperture)
D50_GEO = 1e-9                          # m     (geometric source blur: diameter holding 50% of the current)

# Range each machine setting is drawn from, per sample. Low == high means fixed.
MACHINE_RANGES = {
    'voltage': (VOLTAGE, VOLTAGE),      # V
    'theta':   (THETA, THETA),          # rad
    'd50_geo': (D50_GEO, D50_GEO),      # m
}

#################################################################################################################################################################
"""Image sampling"""

# The stored images are N_PIXELS x N_PIXELS with pixels of PIXEL_SIZE. The pixel size is fixed in
# metres (not set by the zero-pad factor like in WavePSF0.py), so images made with a different
# voltage or opening angle still line up pixel for pixel.
PIXEL_SIZE = 0.3e-9                     # m     (stored image pixel, so the window is 128 * 0.3 = 38.4 nm)
N_PIXELS = 128                          # -     (stored image is N_PIXELS x N_PIXELS)

# The PSF is first simulated on a finer and larger grid, then cropped and binned down:
#   - BIN_FACTOR simulation pixels are averaged into one stored pixel (finer simulation)
#   - PAD_FACTOR makes the simulated field larger than the stored window, so the FFT (which is
#     periodic) does not wrap the edges of the PSF back around into the window
BIN_FACTOR = 2                          # -     (simulation pixels per stored pixel, per direction)
PAD_FACTOR = 2                          # -     (simulated field width / stored window width)
SIM_PIXEL_SIZE = PIXEL_SIZE / BIN_FACTOR        # m     (simulation pixel)
N_SIM = N_PIXELS * BIN_FACTOR * PAD_FACTOR      # -     (simulation grid is N_SIM x N_SIM, here 512)

#################################################################################################################################################################
"""Through-focus planes"""

# Each sample is a stack of PSFs at these defocus offsets, added on top of the sample's own C1.
# These are the input channels of the CNN. A stack is needed because a single in-focus PSF looks
# exactly the same when the signs of all even aberrations (C1, A1, C3, A3) are flipped together;
# the under- and overfocused planes break that ambiguity (see Figures/03_sign_ambiguity.png).
DEFOCUS_OFFSETS = [-0.2e-6, 0.0, 0.2e-6]    # m     (under focus, in focus, over focus)

#################################################################################################################################################################
"""Aberration coefficients"""

# Order of the coefficients everywhere (same order as WavePSF0.py):
#   C1 defocus,  A1 two-fold astigmatism,  B2 axial coma,  A2 three-fold astigmatism,
#   C3 spherical aberration,  A3 four-fold astigmatism   (r = real part, i = imaginary part)
COEFF_NAMES = ['C1', 'A1r', 'A1i', 'B2r', 'B2i', 'A2r', 'A2i', 'C3', 'A3r', 'A3i']

# Range each coefficient is drawn from (uniformly). The limits are chosen so that every
# coefficient on its own gives an in-focus blur of a few nm, i.e. they all change the PSF by a
# similar, clearly visible amount (see Figures/01_one_coefficient_in_focus.png).
COEFF_RANGES = {
    'C1':  (-0.3e-6, 0.3e-6),           # m     (defocus)
    'A1r': (-0.3e-6, 0.3e-6),           # m     (two-fold astigmatism)
    'A1i': (-0.3e-6, 0.3e-6),           # m
    'B2r': (-16e-6, 16e-6),             # m     (axial coma)
    'B2i': (-16e-6, 16e-6),             # m
    'A2r': (-25e-6, 25e-6),             # m     (three-fold astigmatism)
    'A2i': (-25e-6, 25e-6),             # m
    'C3':  (-5e-3, 5e-3),               # m     (spherical aberration)
    'A3r': (-0.5e-3, 0.5e-3),           # m     (four-fold astigmatism)
    'A3i': (-0.5e-3, 0.5e-3),           # m
}

#################################################################################################################################################################
"""Dataset"""

N_SAMPLES = 10000                       # -     (samples in a dataset)
DATASET_NAME = 'psf_v1'                 # -     (folder name inside Data/)
BATCH_SIZE = 64                         # -     (samples computed at once; only affects speed and memory)
MIN_CONTAINED = 0.97                    # -     (a PSF is redrawn if less than this fraction of the beam
                                        #        current lands inside the stored window, in any plane)
SEED = 0                                # -     (random seed, so a dataset can be made again exactly)

#################################################################################################################################################################
"""File locations"""

# All paths are relative to this folder, so the scripts work from wherever they are run.
FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(FOLDER, 'Data')
FIGURES_FOLDER = os.path.join(FOLDER, 'Figures')
ORIGINAL_FILE = os.path.join(FOLDER, 'WavePSF0.py')
