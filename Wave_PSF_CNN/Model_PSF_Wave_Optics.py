"""This file computes wave-optical SEM probe shapes (PSFs) from aberration coefficients.
   It is a batched version of calcPSFOTFwave / ZernikePhase in WavePSF0.py."""

#################################################################################################################################################################
"""Imports"""

# External imports
import math
import numpy as np
from scipy import fft

# Internal imports
from constants import (WAVELENGTH_CONSTANT, EPS_RELATIVISTIC, VOLTAGE, THETA, D50_GEO,
                       SIM_PIXEL_SIZE, N_SIM, N_PIXELS, BIN_FACTOR, DEFOCUS_OFFSETS)

#################################################################################################################################################################
"""Info"""

# How a PSF is made (Fourier optics, the same steps as WavePSF0.py):
#
#   1. Pupil: every electron leaves the aperture at some angle (alpha_x, alpha_y). Only angles
#      inside the aperture pass:  alpha_x^2 + alpha_y^2 <= theta^2.
#   2. Aberrations delay the wave by W(alpha_x, alpha_y) metres, so the pupil function is
#          P = exp(i k W)          with k = 2 pi / lambda
#   3. The probe amplitude on the specimen is the Fourier transform of the pupil, and the
#      intensity PSF is its absolute value squared:   PSF = |FFT(P)|^2
#   4. The finite source size blurs the probe: the PSF is convolved with a Gaussian whose
#      diameter holding 50% of the current is d50_geo (done as a multiplication in Fourier space).
#
# The aberration function (Zernike-Nijboer form, see Born and Wolf / Hawkes and Kasper):
#
#   W = 1/2 C1 (a^2 + b^2)                                       defocus
#     + 1/2 A1r (a^2 - b^2) + A1i a b                            two-fold astigmatism
#     + (B2r a - B2i b)(a^2 + b^2 - 2/3 theta^2)                 axial coma
#     + A2r a (a^2/3 - b^2) + A2i b (a^2 - b^2/3)                three-fold astigmatism
#     + 1/4 C3 (a^2 + b^2)(a^2 + b^2 - theta^2)                  spherical aberration
#     + A3r (a^4 + b^4 - 6 a^2 b^2) + A3i 4 a b (a^2 - b^2)      four-fold astigmatism
#
#   with a = alpha_x and b = alpha_y in rad, all coefficients in m, W in m.
#
# Sampling: on an n x n grid with pixel size dx, the FFT links pupil angles and positions through
#   alpha step = lambda / (n dx)
# so fixing the PSF pixel size dx (instead of the zero-pad factor of WavePSF0.py) keeps the image
# scale the same for any voltage or opening angle. The zero-pad factor is still available through
# zero_pad_factor() for comparing with the original.

#################################################################################################################################################################
"""Functions"""

# Pupil patterns are cached: they only depend on the machine setting, not on the coefficients
_pupil_cache = {}

def electron_wavelength(voltage):
    """
    Relativistic de Broglie wavelength of the electrons (same formula as WavePSF0.py).

    voltage : acceleration voltage (V), a number or an array

    Returns the wavelength (m).
    """
    voltage = np.asarray(voltage, dtype=float)
    return WAVELENGTH_CONSTANT / np.sqrt(voltage + EPS_RELATIVISTIC * voltage ** 2)

def zero_pad_factor(voltage=VOLTAGE, theta=THETA, pixel_size=SIM_PIXEL_SIZE):
    """
    The zero-pad factor (zpf) of WavePSF0.py that gives this pixel size, from
    pixel_size = 0.5 * lambda / (zpf * theta). Only needed to compare with the original code.

    voltage    : acceleration voltage (V)
    theta      : half opening angle (rad)
    pixel_size : PSF pixel size (m)

    Returns zpf (-).
    """
    return 0.5 * float(electron_wavelength(voltage)) / (pixel_size * theta)

def aberration_function(alpha_x, alpha_y, theta, coeffs):
    """
    Wavefront aberration W at the pupil angles (alpha_x, alpha_y), see the Info section.

    alpha_x, alpha_y : pupil angles (rad), arrays of the same shape
    theta            : half opening angle (rad)
    coeffs           : the 10 coefficients (m) in the order of COEFF_NAMES

    Returns W (m), same shape as alpha_x.
    """
    C1, A1r, A1i, B2r, B2i, A2r, A2i, C3, A3r, A3i = coeffs
    a2 = alpha_x ** 2
    b2 = alpha_y ** 2
    r2 = a2 + b2
    th2 = theta ** 2

    W = 0.5 * C1 * r2                                                           # defocus
    W += 0.5 * A1r * (a2 - b2) + A1i * alpha_x * alpha_y                        # two-fold astigmatism
    W += (B2r * alpha_x - B2i * alpha_y) * (r2 - (2 / 3) * th2)                 # axial coma
    W += A2r * alpha_x * (a2 / 3 - b2) + A2i * alpha_y * (a2 - b2 / 3)          # three-fold astigmatism
    W += 0.25 * C3 * r2 * (r2 - th2)                                            # spherical aberration
    W += A3r * (a2 ** 2 + b2 ** 2 - 6 * a2 * b2) + A3i * 4 * alpha_x * alpha_y * (a2 - b2)  # four-fold astigmatism
    return W

def _pupil_basis(n, pixel_size, voltage, theta):
    """
    The pixels inside the aperture, and the 10 aberration patterns evaluated on them.

    W is linear in the coefficients (W = C1 * pattern_C1 + A1r * pattern_A1r + ...), so the
    patterns are computed once per machine setting and reused for every sample. Only the pixels
    inside the aperture are kept (about 15% of the grid), which makes the generation much faster.

    n          : simulation grid size (-)
    pixel_size : PSF pixel size (m)
    voltage    : acceleration voltage (V)
    theta      : half opening angle (rad)

    Returns (inside, patterns, k):
      inside   : flat indices of the grid pixels inside the aperture
      patterns : array (10, number of inside pixels), W of each coefficient set to 1 m
      k        : wave number 2 pi / lambda (1/m)
    """
    key = (n, pixel_size, voltage, theta)
    if key not in _pupil_cache:
        wavelength = float(electron_wavelength(voltage))
        alpha_step = wavelength / (n * pixel_size)              # rad per pupil pixel
        u = np.arange(n) - n // 2                               # pixel index, 0 at the centre
        # Rows are y and columns are x, as ZernikePhase(j-xc, i-xc) in WavePSF0.py
        alpha_y, alpha_x = np.meshgrid(u * alpha_step, u * alpha_step, indexing='ij')
        in_aperture = alpha_x ** 2 + alpha_y ** 2 <= theta ** 2

        patterns = np.zeros((10, in_aperture.sum()))
        for i in range(10):
            unit = np.zeros(10)
            unit[i] = 1.0                                       # only coefficient i switched on
            patterns[i] = aberration_function(alpha_x[in_aperture], alpha_y[in_aperture], theta, unit)

        _pupil_cache.clear()                                    # keep only the latest setting in memory
        _pupil_cache[key] = (np.flatnonzero(in_aperture), patterns, 2 * math.pi / wavelength)
    return _pupil_cache[key]

def psf_field(coeffs, n, pixel_size, voltage, theta, d50_geo):
    """
    Full simulated PSF for a batch of coefficient sets (steps 1-4 of the Info section).
    The origin is at pixel [0, 0], not in the centre, just like calcPSFOTFwave.

    coeffs     : array (B, 10), coefficients (m) of B samples
    n          : simulation grid size (-)
    pixel_size : PSF pixel size (m)
    voltage    : acceleration voltage (V), one number or one per sample
    theta      : half opening angle (rad), one number or one per sample
    d50_geo    : source blur (m), one number or one per sample

    Returns array (B, n, n), every PSF normalised to a sum of 1.
    """
    coeffs = np.atleast_2d(np.asarray(coeffs, dtype=float))
    B = len(coeffs)
    voltage = np.broadcast_to(np.asarray(voltage, dtype=float), (B,))
    theta = np.broadcast_to(np.asarray(theta, dtype=float), (B,))
    d50_geo = np.broadcast_to(np.asarray(d50_geo, dtype=float), (B,))

    # Steps 1 and 2: pupil function exp(i k W), zero outside the aperture.
    # Samples are handled per machine setting, since the aperture and patterns depend on it.
    pupil = np.zeros((B, n * n), dtype=np.complex64)
    for v, t in set(zip(voltage, theta)):
        rows = np.flatnonzero((voltage == v) & (theta == t))
        inside, patterns, k = _pupil_basis(n, pixel_size, v, t)
        # Phase in rad; taken modulo 2 pi before going to single precision, so no accuracy is lost
        phase = np.mod(k * (coeffs[rows] @ patterns), 2 * math.pi).astype(np.float32)
        pupil[rows[:, None], inside[None, :]] = np.cos(phase) + 1j * np.sin(phase)

    # Step 3: probe intensity = |Fourier transform of the pupil|^2
    psf = np.abs(fft.fft2(pupil.reshape(B, n, n), workers=-1)) ** 2
    psf /= psf.sum(axis=(-2, -1), keepdims=True)

    # Step 4: source blur. A Gaussian with standard deviation sigma = d50 / sqrt(ln 256) (in pixels),
    # applied by multiplying the Fourier transform of the PSF by exp(-2 pi^2 sigma^2 f^2).
    # rfft2 is used because the PSF is real, which halves the work.
    sigma_px = (d50_geo / (pixel_size * math.sqrt(math.log(256.0)))).reshape(B, 1, 1)
    if np.any(sigma_px > 0):
        f_y = fft.fftfreq(n)                                    # spatial frequency (1/pixel) along y
        f_x = fft.rfftfreq(n)                                   # ... and along x (half, since rfft2)
        f2 = (f_y[:, None] ** 2 + f_x[None, :] ** 2)[None]
        gaussian = np.exp(-2 * math.pi ** 2 * sigma_px ** 2 * f2).astype(np.float32)
        psf = np.abs(fft.irfft2(fft.rfft2(psf, workers=-1) * gaussian, s=(n, n), workers=-1))
        psf /= psf.sum(axis=(-2, -1), keepdims=True)
    return psf.astype(np.float32)

def _repeat_per_plane(value, B, P):
    """
    Turn one machine setting (a number, or one value per sample) into one value per plane,
    in the same order as the flattened (sample, plane) batch in psf_stack.

    value : a number or array (B,)
    B     : number of samples
    P     : number of planes per sample

    Returns array (B * P,).
    """
    return np.repeat(np.broadcast_to(np.asarray(value, dtype=float), (B,)), P)

def psf_stack(coeffs, voltage=VOLTAGE, theta=THETA, d50_geo=D50_GEO, through_focus=True):
    """
    PSFs in the stored format: a through-focus stack per sample, cropped to the stored window
    and binned down to N_PIXELS x N_PIXELS pixels of PIXEL_SIZE.

    coeffs        : array (B, 10), coefficients (m)
    voltage       : acceleration voltage (V), one number or one per sample
    theta         : half opening angle (rad), one number or one per sample
    d50_geo       : source blur (m), one number or one per sample
    through_focus : True gives one plane per DEFOCUS_OFFSETS, False only the sample's own focus

    Returns (images, contained):
      images    : array (B, planes, N_PIXELS, N_PIXELS), fraction of the beam current per pixel
      contained : array (B, planes), fraction of the beam current inside the stored window
                  (1 would mean nothing falls outside)
    """
    coeffs = np.atleast_2d(np.asarray(coeffs, dtype=float))
    B = len(coeffs)
    offsets = np.array(DEFOCUS_OFFSETS if through_focus else [0.0])
    P = len(offsets)

    # One coefficient set per plane: the same sample with C1 shifted by each defocus offset
    plane_coeffs = np.repeat(coeffs[:, None, :], P, axis=1)
    plane_coeffs[:, :, 0] += offsets
    field = psf_field(plane_coeffs.reshape(B * P, 10), N_SIM, SIM_PIXEL_SIZE,
                      _repeat_per_plane(voltage, B, P), _repeat_per_plane(theta, B, P),
                      _repeat_per_plane(d50_geo, B, P))

    # Move the origin to the centre, cut out the stored window and add up BIN_FACTOR x BIN_FACTOR blocks
    field = fft.fftshift(field, axes=(-2, -1))
    window = N_PIXELS * BIN_FACTOR                              # simulation pixels in the stored window
    start = N_SIM // 2 - window // 2
    crop = field[:, start:start + window, start:start + window]
    images = crop.reshape(B * P, N_PIXELS, BIN_FACTOR, N_PIXELS, BIN_FACTOR).sum(axis=(2, 4))
    contained = images.sum(axis=(-2, -1))
    return images.reshape(B, P, N_PIXELS, N_PIXELS), contained.reshape(B, P)
