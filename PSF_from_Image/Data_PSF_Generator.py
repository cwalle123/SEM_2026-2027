# External imports
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from matplotlib.colors import LogNorm
import csv
from pathlib import Path

# Internal imports

##################################################################################################################################################################################################################
"""Constants"""

# Example abberation coefficients for testing the PSF generation
c1  = 0.5e-6                #defocus [meter]        (e-6)
a1r = 0;      a1i = 0;      #astigmatism            (e-8)
b2r = 0;      b2i = 0;      #coma                   (e-6)
a2r = 0;      a2i = 10e-6;  #3-fold astigmatism     (e-6)
a3r = 0;      a3i = 0;      #4-fold astigmatism     (e-4)
c3  = 0                     #spherical abberation   (e-3)
example_abberation_vector = np.array([c1, a1r, a1i, b2r, b2i, a2r, a2i, c3, a3r, a3i])   # order: C3 comes before A3

electron_charge = 1.602176634e-19       # [C]
V = 30000                               # Voltage [V]
h = 6.62607015e-34                      # Planck's constant [J*s]
c = 299792458                           # Speed of light [m/s]
electron_mass = 9.10938356e-31          # Electron mass [kg]
electron_wavelength = h / np.sqrt(2 * electron_mass * electron_charge * V + (1 / c**2 ) * (electron_charge * V)**2)  # Relativistic electron wavelength [m] for an accelerating voltage [V]

##################################################################################################################################################################################################################
"""Functions"""

# ---- Functions to generate EXAMPLE surfaces and PSFs, and to convolve them. THESE ARE ONLY USED FOR CHECKING THAT THE PSF IS WORKING -----"
def make_sample_surface(n=201, extent=10):
    """Sinusoidal 2D surface with fine detail, so blurring by the PSF is visible."""
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(x, y)
    S = (0.3 * np.sin(0.5 * X) + 0.2 * np.cos(0.4 * Y) + 0.1 * np.sin(0.3 * (X + Y)) + 0.1 * np.sin(4 * X) * np.cos(4 * Y))
    return X, Y, S

def plot_example_surface(x, y, surface):
    """This function plots the example surface"""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, surface, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_zlim(-3, 3)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.show()

# plot_example_surface(*make_sample_surface())

def psf_grid(size):
    """Centered integer grid, size must be odd."""
    r = np.arange(size) - size // 2
    return np.meshgrid(r, r)

def make_astigmatic_psf(size=31, sigma_x=5, sigma_y=3):
    """2D astigmatic Gaussian PSF, normalized to sum 1."""
    X, Y = psf_grid(size)
    g = np.exp(-(X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2)))
    return g / g.sum()

def convolve(f, g):
    """h = f * g, output same shape as f."""
    return fftconvolve(f, g, mode="same")

def plot_convolution():
    """This function plots the convolution of f and g."""
    f = make_sample_surface()[2]
    g = make_astigmatic_psf(size=31, sigma_x=5, sigma_y=3)     # size >= 6*max(sigma)+1, otherwise the Gaussian tails get cut off
    h = convolve(f, g)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(*np.meshgrid(np.arange(f.shape[0]), np.arange(f.shape[1])), f, cmap="viridis", linewidth=0, antialiased=True)
    ax1.set_title("f (input surface)")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(*np.meshgrid(np.arange(g.shape[0]), np.arange(g.shape[1])), g, cmap="viridis", linewidth=0, antialiased=True)
    ax2.set_title("g (PSF)")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(*np.meshgrid(np.arange(h.shape[0]), np.arange(h.shape[1])), h, cmap="viridis", linewidth=0, antialiased=True)
    ax3.set_title("h = f * g (convolution)")

    for ax in (ax1, ax2, ax3):
        ax.set_zlim(-3, 3)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    plt.tight_layout()
    plt.show()

# plot_convolution()

# ----- Functions to generate PSFs from abberation coefficients. These will be used later in a loop to generate a dataset of PSFs -----

def abberation_function(tx, ty, alpha, aberration_vector):
    """phi(x', y') [m]: Zernike-balanced wave aberration over a circular aperture.
    tx, ty = ray angles [rad], alpha = half opening angle [rad].
    aberration_vector = [C1, A1r, A1i, B2r, B2i, A2r, A2i, C3, A3r, A3i] in meters."""
    C1  = aberration_vector[0]
    A1r = aberration_vector[1];    A1i = aberration_vector[2]
    B2r = aberration_vector[3];    B2i = aberration_vector[4]
    A2r = aberration_vector[5];    A2i = aberration_vector[6]
    C3  = aberration_vector[7]
    A3r = aberration_vector[8];    A3i = aberration_vector[9]

    tx2 = tx**2;   ty2 = ty**2;   al2 = alpha**2
    S  = 0.5*C1*(tx2 + ty2) + 0.5*A1r*(tx2 - ty2) + A1i*tx*ty                    # defocus, 2-fold astigmatism:  Re{1/2 w w* C1 + 1/2 w*^2 A1}
    S += (B2r*tx - B2i*ty)*(tx2 + ty2 - (2/3)*al2)                               # coma (tilt-balanced):          Re{w^2 w* B2}
    S += A2r*tx*((tx2/3) - ty2) + A2i*ty*(tx2 - (ty2/3))                         # 3-fold astigmatism:            Re{1/3 w*^3 A2}
    S += 0.25*C3*(tx2 + ty2)*(tx2 + ty2 - al2)                                   # spherical (defocus-balanced):  Re{1/4 (w w*)^2 C3}
    S += 0.25*(A3r*(tx**4 + ty**4 - 6*tx2*ty2) + A3i*(4*tx*ty*(tx2 - ty2)))      # 4-fold astigmatism:            Re{1/4 w*^4 A3}, same 1/n prefactor pattern as A1 and A2
    return S

def pupil_function(n, pixel_size, wavelength, alpha, aber):
    """P(x', y') = A(x', y') * exp(j 2pi/lambda * phi(x', y'))   (Eq. 3 and 4)."""
    dtheta = wavelength / (n * pixel_size)      # angular step, chosen so that the FFT lands on image pixels of size pixel_size
    if n * dtheta / 2 < alpha:
        raise ValueError(f"Aperture doesn't fit: need pixel_size < {wavelength / (2 * alpha):.3e} m")
    theta = (np.arange(n) - n // 2) * dtheta
    TX, TY = np.meshgrid(theta, theta)
    A = (np.hypot(TX, TY) <= alpha).astype(float)
    phi = abberation_function(TX, TY, alpha, aber)
    P = A * np.exp(1j * 2 * np.pi / wavelength * phi)

    phase = 2 * np.pi / wavelength * phi
    gy, gx = np.gradient(phase)
    max_step = np.max(np.hypot(gx, gy)[A > 0])
    if max_step > np.pi:
        warnings.warn(f"Pupil phase undersampled ({max_step:.1f} rad/px > pi): increase n")
    return theta, A, phase, P

def make_aberration_psf(aber, voltage=V, alpha=20e-3, pixel_size=20e-12, n=2048):
    """Aberration coefficients [m] -> intensity PSF Psi(x, y), normalized to sum 1."""
    def electron_wavelength(V):
        return h / np.sqrt(2 * electron_mass * electron_charge * V + (1 / c**2 ) * (electron_charge * V)**2)
    
    wavelength = electron_wavelength(voltage)
    theta, A, phase, P = pupil_function(n, pixel_size, wavelength, alpha, aber)

    # Eq. 5: the complex PSF U(x, y) is the Fourier transform of the pupil function P(x', y')
    U = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(P)))
    # Intensity PSF Psi(x, y) = |U(x, y)|^2, normalized to sum 1 (replaces the 1/(pi theta^2) prefactor) so it can be used as a convolution kernel
    psi = np.abs(U)**2
    psi /= psi.sum()

    x = (np.arange(n) - n // 2) * pixel_size
    phase_masked = np.where(A > 0, phase, np.nan)
    return theta, x, phase_masked, psi

def plot_psf(theta, x, phase, psi, alpha, crop=600):
    """This function plots the pupil phase and the PSF intensity (linear and log scale).
    crop = half width of the PSF window around the center [px]."""
    c = len(x) // 2
    s = slice(c - crop, c + crop)                                   # PSF window around the center
    a = np.abs(theta) <= 1.1 * alpha                                # only the angles around the aperture
    x_nm = x[s] * 1e9
    th_mrad = theta[a] * 1e3
    psf_extent = [x_nm[0], x_nm[-1], x_nm[0], x_nm[-1]]
    pupil_extent = [th_mrad[0], th_mrad[-1], th_mrad[0], th_mrad[-1]]

    wrapped_phase = np.angle(np.exp(1j * phase[np.ix_(a, a)]))      # wrap to [-pi, pi], NaN outside the aperture stays NaN
    psi_crop = psi[s, s]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.2))

    im1 = ax1.imshow(wrapped_phase, origin="lower", extent=pupil_extent, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax1.add_patch(plt.Circle((0, 0), alpha * 1e3, fill=False, color="k", lw=1))
    ax1.set_title("Pupil phase  2π/λ·φ(x', y')")
    ax1.set_xlabel("θx [mrad]"); ax1.set_ylabel("θy [mrad]")
    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.85, label="Phase [rad]")
    cb1.set_ticks([-np.pi, 0, np.pi]); cb1.set_ticklabels(["−π", "0", "π"])

    im2 = ax2.imshow(psi_crop, origin="lower", extent=psf_extent, cmap="inferno")
    ax2.set_title("PSF intensity Ψ(x, y)")
    ax2.set_xlabel("x [nm]"); ax2.set_ylabel("y [nm]")
    fig.colorbar(im2, ax=ax2, shrink=0.85, label="Ψ (normalized)")

    im3 = ax3.imshow(psi_crop, origin="lower", extent=psf_extent, cmap="inferno",
                     norm=LogNorm(vmin=psi_crop.max() * 1e-4, vmax=psi_crop.max()))
    ax3.set_title("PSF intensity Ψ(x, y), log scale")
    ax3.set_xlabel("x [nm]"); ax3.set_ylabel("y [nm]")
    fig.colorbar(im3, ax=ax3, shrink=0.85, label="Ψ (normalized)")

    for ax in (ax1, ax2, ax3):
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()

# alpha = 20e-3
# theta, x, phase, psi = make_aberration_psf(example_abberation_vector, voltage=V, alpha=alpha, pixel_size=20e-12, n=2048)
# plot_psf(theta, x, phase, psi, alpha, crop=600)

# ----- Functions to generate a TRAINING dataset of PSFs with their abberation vectors -----

aberration_labels = ["C1", "A1r", "A1i", "B2r", "B2i", "A2r", "A2i", "C3", "A3r", "A3i"]
aberration_ranges = np.array([0.5e-6,                   # C1        max |value| per coefficient [m], vectors are sampled uniformly in [-range, range]
                              1e-8,   1e-8,             # A1r, A1i
                              1e-6,   1e-6,             # B2r, B2i
                              10e-6,  10e-6,            # A2r, A2i
                              1e-3,                     # C3
                              1e-4,   1e-4])            # A3r, A3i

def random_aberration_vectors(n_samples, ranges=aberration_ranges, seed=None):
    """Random aberration vectors, each coefficient uniform in [-range, range]. Shape (n_samples, 10)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(n_samples, len(ranges))) * ranges

def bin_image(image, out_size):
    """Downsamples a square image to (out_size, out_size) by summing blocks of pixels (keeps the total intensity)."""
    factor = image.shape[0] // out_size
    return image[:out_size * factor, :out_size * factor].reshape(out_size, factor, out_size, factor).sum(axis=(1, 3))

def generate_training_dataset(n_samples, out_size=128, crop=640, voltage=V, alpha=20e-3, pixel_size=20e-12, n=2048,
                              ranges=aberration_ranges, val_fraction=0.1, test_fraction=0.1, seed=0,
                              filename="psf_dataset.npz", folder="Data", n_preview=10):
    """Generates n_samples PSFs from random aberration vectors and saves everything in one file in the Data folder:
      images              (N, out_size, out_size) float32  -> PSF images, binned from the (2*crop)^2 center of the full PSF
      labels              (N, 10) float32                  -> aberration vectors scaled to [-1, 1] (= aberration_vectors / ranges), what the network predicts
      aberration_vectors  (N, 10)                          -> the real coefficients [m]
      ranges, label_names                                  -> to convert predicted labels back to meters: labels * ranges
      train_idx, val_idx, test_idx                         -> fixed split, so every training run uses the same test set
      voltage, alpha, pixel_size, binned_pixel_size        -> microscope settings the images were made with
    The first n_preview images are also saved as PNG for a quick look."""
    folder = Path(__file__).parent / folder
    folder.mkdir(exist_ok=True)
    if (2 * crop) % out_size != 0 or crop > n // 2:
        raise ValueError("2*crop must be divisible by out_size and crop must be <= n // 2")
    s = slice(n // 2 - crop, n // 2 + crop)

    aberration_vectors = random_aberration_vectors(n_samples, ranges, seed)
    images = np.empty((n_samples, out_size, out_size), dtype=np.float32)

    for i, aber in enumerate(aberration_vectors):
        psi = make_aberration_psf(aber, voltage=voltage, alpha=alpha, pixel_size=pixel_size, n=n)[3][s, s]
        images[i] = bin_image(psi, out_size)
        if i < n_preview:
            plt.imsave(folder / f"preview_{i:05d}.png", images[i], cmap="inferno", origin="lower")
        if (i + 1) % 100 == 0 or i + 1 == n_samples:
            print(f"Generated {i + 1}/{n_samples} PSFs")

    labels = (aberration_vectors / ranges).astype(np.float32)          # all coefficients on the same [-1, 1] scale, otherwise C3 (e-3) dominates the loss over A1 (e-8)

    idx = np.random.default_rng(seed + 1).permutation(n_samples)
    n_test = int(test_fraction * n_samples)
    n_val = int(val_fraction * n_samples)
    test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]

    np.savez_compressed(folder / filename, images=images, labels=labels, aberration_vectors=aberration_vectors,
                        ranges=ranges, label_names=np.array(aberration_labels),
                        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                        voltage=voltage, alpha=alpha, pixel_size=pixel_size,
                        binned_pixel_size=pixel_size * (2 * crop // out_size))
    print(f"Saved {n_samples} samples to {folder / filename}")

#########################################################################################################
""""Run this file"""

def main():
    """This functions runs everything from this file"""
    # plot_example_surface(example_sample_surface)

    # generate_training_dataset(n_samples=50) # generates 50 PSFs and saves them in Data/psf_dataset.npz, also saves the first 10 as PNG for a quick look

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else