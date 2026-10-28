# External imports
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image   # Pillow (installed with matplotlib), used to save 16-bit PNG images

# Internal imports

#########################################################################################################
"""Constants"""

# Default simulation settings (used by main(), can be overridden in generate_psf())
N_PIXELS   = 1024    # PSF grid size: N_PIXELS x N_PIXELS
ZERO_PAD   = 5       # zero-padding factor (>= 1, sensible up to ~15). Larger = finer PSF pixels
HALF_ANGLE = 0.015   # beam half opening angle (convergence semi-angle) [rad]
VOLTAGE    = 30000   # accelerating voltage [V]
D50_GEO    = 1e-9    # geometric probe size (source image): diameter containing 50% of the current [m]

# Relativistic electron wavelength: lambda = 1.226e-9 / sqrt(U * (1 + 9.785e-7 * U)), with U in volts
WAVELENGTH_CONST  = 1.226e-9   # [m * V^0.5]
RELATIVISTIC_CORR = 9.785e-7   # [1/V]

# Training data settings
N_SAMPLES   = 1000                                                   # number of PSFs to generate
DATA_FOLDER = Path(__file__).resolve().parent / "Data" / "PSF_data"  # output folder, next to this file
LABEL_NAMES = ["C1", "A1r", "A1i", "B2r", "B2i", "A2r", "A2i", "C3", "A3r", "A3i"]  # order of the aberration vector

# Maximum size of each aberration when drawing random training data [m]
# (based on the typical values noted in WavePSF0.py, adjust to match your microscope)
ABERRATION_RANGES = {"C1": 0.5e-6,   # defocus
                     "A1": 1e-8,     # 2-fold astigmatism
                     "B2": 1e-6,     # axial coma
                     "A2": 10e-6,    # 3-fold astigmatism
                     "C3": 1e-3,     # spherical aberration
                     "A3": 1e-4}     # 4-fold astigmatism

#########################################################################################################
"""Functions"""

def electron_wavelength(voltage):
    """
    Compute the relativistic wavelength of an electron.

    Args:
        voltage (float): accelerating voltage [V].

    Returns:
        float: electron wavelength [m].
    """
    corrected_voltage = voltage * (1 + RELATIVISTIC_CORR * voltage)
    return WAVELENGTH_CONST / np.sqrt(corrected_voltage)


def make_aberration_vector(C1=0.0, A1=0j, B2=0j, A2=0j, C3=0.0, A3=0j):
    """
    Pack the 6 aberration coefficients into a single numpy vector.

    Non-round aberrations have a size and a direction, so they are given as complex numbers
    (real part = x component, imaginary part = y component), e.g. A1 = 5e-9 + 2e-9j.

    Args:
        C1 (float):   defocus [m].
        A1 (complex): two-fold astigmatism [m].
        B2 (complex): axial coma [m].
        A2 (complex): three-fold astigmatism [m].
        C3 (float):   spherical aberration, Cs [m].
        A3 (complex): four-fold astigmatism [m].

    Returns:
        np.ndarray: length-10 float vector
                    [C1, A1r, A1i, B2r, B2i, A2r, A2i, C3, A3r, A3i]
                    (same ordering as the `aber` array in WavePSF0.py).
    """
    A1, B2, A2, A3 = complex(A1), complex(B2), complex(A2), complex(A3)
    return np.array([C1,
                     A1.real, A1.imag,
                     B2.real, B2.imag,
                     A2.real, A2.imag,
                     C3,
                     A3.real, A3.imag])


def aberration_function(ax, ay, theta, aber):
    """
    Compute the wave aberration W (the optical path difference) at each beam angle.

    The coma and spherical terms are balanced against lower orders in the same way as
    Zernike polynomials (see Born & Wolf, Principles of Optics), exactly as in WavePSF0.py.

    Args:
        ax (np.ndarray): beam angle in x [rad].
        ay (np.ndarray): beam angle in y [rad].
        theta (float):   half opening angle, i.e. the angle at the aperture edge [rad].
        aber (np.ndarray): aberration vector from make_aberration_vector().

    Returns:
        np.ndarray: wave aberration W(ax, ay) [m], same shape as ax.
    """
    C1, A1r, A1i, B2r, B2i, A2r, A2i, C3, A3r, A3i = aber
    r2  = ax**2 + ay**2   # squared angle from the optical axis
    th2 = theta**2

    W  = 0.5 * C1 * r2                                                        # defocus
    W += 0.5 * A1r * (ax**2 - ay**2) + A1i * ax * ay                          # 2-fold astigmatism
    W += (B2r * ax - B2i * ay) * (r2 - (2/3) * th2)                           # axial coma
    W += A2r * ax * (ax**2 / 3 - ay**2) + A2i * ay * (ax**2 - ay**2 / 3)      # 3-fold astigmatism
    W += 0.25 * C3 * r2 * (r2 - th2)                                          # spherical aberration
    W += A3r * (ax**4 + ay**4 - 6 * ax**2 * ay**2) + A3i * 4 * ax * ay * (ax**2 - ay**2)  # 4-fold astigmatism
    return W


def generate_psf(aber, n=N_PIXELS, zpf=ZERO_PAD, voltage=VOLTAGE, theta=HALF_ANGLE, d50geo=D50_GEO):
    """
    Simulate the wave-optical probe PSF of an SEM (chromatic aberration not included).

    Steps:
        1. Build the pupil (aperture) function: a disc with phase exp(i * k * W).
        2. Fourier transform it to get the probe wavefunction; its intensity is the PSF.
        3. Blur with a Gaussian to account for the finite source size (d50geo).

    Args:
        aber (np.ndarray): aberration vector from make_aberration_vector().
        n (int):           PSF grid size (n x n pixels).
        zpf (float):       zero-padding factor (>= 1). Sets the PSF pixel size.
        voltage (float):   accelerating voltage [V].
        theta (float):     half opening angle [rad].
        d50geo (float):    geometric probe size (50% diameter) [m]. Use 0 for no source blur.

    Returns:
        psf (np.ndarray):   n x n PSF, centred in the array, normalised to a peak of 1.
        otf (np.ndarray):   n x n complex OTF, zero frequency at [0, 0], normalised so otf[0, 0] = 1.
        pixel_size (float): size of one PSF pixel [m].
    """
    assert zpf >= 1, "zero pad factor must be >= 1"

    # Basic optics quantities
    wavelength = electron_wavelength(voltage)
    k = 2 * np.pi / wavelength                    # wave number [1/m]
    pixel_size = wavelength / (2 * zpf * theta)   # PSF pixel size [m]

    # Pupil-plane grid: the aperture is a disc of radius n/(2*zpf) pixels in the middle of the grid,
    # the rest is zeros ("zero padding"), which is what sets the PSF pixel size above
    centre = n // 2
    aperture_radius_px = n / (2 * zpf)
    angle_per_px = theta / aperture_radius_px     # beam angle covered by one pupil pixel [rad]
    rows, cols = np.indices((n, n))
    x_px = cols - centre
    y_px = rows - centre
    inside_aperture = x_px**2 + y_px**2 <= aperture_radius_px**2

    # Pupil function: unit amplitude inside the aperture, phase = k * W
    W = aberration_function(x_px * angle_per_px, y_px * angle_per_px, theta, aber)
    pupil = np.where(inside_aperture, np.exp(1j * k * W), 0)

    # Probe wavefunction = Fourier transform of the pupil, PSF = its intensity
    psf = np.abs(np.fft.fft2(pupil))**2
    psf /= psf.sum()
    otf = np.fft.fft2(psf)

    # Finite source size: convolving the PSF with a Gaussian = multiplying the OTF by a Gaussian
    if d50geo > 0:
        # A 2D Gaussian with standard deviation sigma has 50% of its intensity within d50 = sigma * sqrt(8 ln 2)
        sigma_px = d50geo / (pixel_size * np.sqrt(8 * np.log(2)))
        f = np.fft.fftfreq(n)                     # spatial frequencies [cycles/pixel], zero at index 0
        f2 = f[:, None]**2 + f[None, :]**2
        otf *= np.exp(-2 * np.pi**2 * sigma_px**2 * f2)
        psf = np.abs(np.fft.ifft2(otf))

    # Move the PSF peak from [0, 0] to the centre of the array and normalise to a peak of 1
    psf = np.fft.fftshift(psf) / psf.max()
    return psf, otf, pixel_size


def plot_psf(psf, pixel_size, fov_nm=None, title="Simulated PSF"):
    """
    Show a PSF as an image together with its horizontal and vertical cross-sections.

    Args:
        psf (np.ndarray):   centred PSF from generate_psf().
        pixel_size (float): PSF pixel size [m].
        fov_nm (float):     width of the region to show [nm]. None shows the full PSF.
        title (str):        figure title.
    """
    n = psf.shape[0]
    centre = n // 2
    coords_nm = (np.arange(n) - centre) * pixel_size * 1e9   # pixel positions [nm], 0 at the centre
    half_width = coords_nm[-1] if fov_nm is None else fov_nm / 2

    fig, (ax_img, ax_cut) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(title)

    # PSF image
    extent = [coords_nm[0], coords_nm[-1], coords_nm[-1], coords_nm[0]]
    im = ax_img.imshow(psf, cmap="gray", extent=extent)
    ax_img.set_xlim(-half_width, half_width)
    ax_img.set_ylim(half_width, -half_width)
    ax_img.set_xlabel("x [nm]")
    ax_img.set_ylabel("y [nm]")
    fig.colorbar(im, ax=ax_img, fraction=0.046)

    # Cross-sections through the centre
    ax_cut.plot(coords_nm, psf[centre, :], "b-", label="x (horizontal)")
    ax_cut.plot(coords_nm, psf[:, centre], "r-", label="y (vertical)")
    ax_cut.set_xlim(-half_width, half_width)
    ax_cut.set_xlabel("position [nm]")
    ax_cut.set_ylabel("intensity (peak = 1)")
    ax_cut.legend()

    plt.tight_layout()
    plt.show()


def random_aberration_vector(rng, ranges=ABERRATION_RANGES):
    """
    Draw a random aberration vector containing all 6 aberrations.

    Each component is drawn uniformly from [-max, +max], except spherical aberration (C3),
    which is drawn from [0, max] because Cs of an uncorrected lens is always positive.

    Args:
        rng (np.random.Generator): random number generator, e.g. np.random.default_rng(seed).
        ranges (dict): maximum size of each aberration [m], see ABERRATION_RANGES.

    Returns:
        np.ndarray: length-10 aberration vector (see make_aberration_vector()).
    """
    def random_complex(max_size):
        """Random complex number with real and imaginary parts uniform in [-max_size, max_size]."""
        return complex(rng.uniform(-max_size, max_size), rng.uniform(-max_size, max_size))

    return make_aberration_vector(C1=rng.uniform(-ranges["C1"], ranges["C1"]),
                                  A1=random_complex(ranges["A1"]),
                                  B2=random_complex(ranges["B2"]),
                                  A2=random_complex(ranges["A2"]),
                                  C3=rng.uniform(0, ranges["C3"]),
                                  A3=random_complex(ranges["A3"]))


def save_psf_dataset(n_samples=N_SAMPLES, folder=DATA_FOLDER, image_size=None, seed=None, **psf_settings):
    """
    Generate random PSFs with generate_psf() and save them as a CNN training set.

    Output (in `folder`):
        psf_00000.png, psf_00001.png, ...  16-bit greyscale PSF images (peak = 65535), the CNN inputs.
        labels.csv          one row per image: filename + its 10-number aberration vector [m], the CNN targets.
        dataset_info.json   the simulation settings used, so the dataset can be reproduced.
    Existing files with the same names are overwritten.

    Args:
        n_samples (int):      number of PSFs to generate.
        folder (str or Path): output folder (created if it doesn't exist).
        image_size (int):     if given, keep only the central image_size x image_size pixels of each PSF.
                              None keeps the full n x n PSF.
        seed (int):           random seed for a reproducible dataset. None gives a different dataset every run.
        **psf_settings:       settings passed on to generate_psf() (n, zpf, voltage, theta, d50geo).
                              Anything not given uses the defaults from the Constants section.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Fill in default settings for anything not given
    settings = dict(n=N_PIXELS, zpf=ZERO_PAD, voltage=VOLTAGE, theta=HALF_ANGLE, d50geo=D50_GEO)
    settings.update(psf_settings)

    with open(folder / "labels.csv", "w") as labels_file:
        labels_file.write("filename," + ",".join(LABEL_NAMES) + "\n")

        for i in range(n_samples):
            # Random aberrations -> PSF
            aber = random_aberration_vector(rng)
            psf, _, pixel_size = generate_psf(aber, **settings)

            # Optionally crop the central region (the PSF peak stays in the middle)
            if image_size is not None:
                start = settings["n"] // 2 - image_size // 2
                psf = psf[start:start + image_size, start:start + image_size]

            # Save the image as 16-bit PNG (0..1 -> 0..65535) and its vector as a row of labels.csv
            filename = f"psf_{i:05d}.png"
            Image.fromarray(np.round(psf * 65535).astype(np.uint16)).save(folder / filename)
            labels_file.write(filename + "," + ",".join(f"{value:.6e}" for value in aber) + "\n")

            if (i + 1) % max(1, n_samples // 10) == 0:
                print(f"saved {i + 1}/{n_samples} PSFs")

    # Record how the dataset was made
    info = dict(n_samples=n_samples, seed=seed, image_size=image_size, pixel_size_m=pixel_size,
                psf_settings=settings, label_names=LABEL_NAMES, aberration_ranges_m=ABERRATION_RANGES)
    with open(folder / "dataset_info.json", "w") as info_file:
        json.dump(info, info_file, indent=4)

    print(f"Dataset saved to {folder}")

#########################################################################################################
""""Run this file"""

def main():
    """This functions runs everything from this file"""
    # 1. Preview a single PSF, using the same example aberrations as WavePSF0.py
    aber = make_aberration_vector(C1=0.5e-6,    # defocus [m]
                                  A1=0,         # 2-fold astigmatism [m] (typ. 1e-8)
                                  B2=0,         # coma [m] (typ. 1e-6)
                                  A2=10e-6j,    # 3-fold astigmatism [m] (typ. 1e-6)
                                  C3=0,         # spherical aberration [m] (typ. 1e-3)
                                  A3=0)         # 4-fold astigmatism [m] (typ. 1e-4)

    psf, otf, pixel_size = generate_psf(aber)
    print(f"PSF: pixel size = {pixel_size * 1e9:.3f} nm")

    plot_psf(psf, pixel_size)   # close the plot window to continue

    # 2. Generate the CNN training set in Data/PSF_data
    save_psf_dataset(N_SAMPLES, seed=0)

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
