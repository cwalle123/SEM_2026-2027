# External imports
import sys
import warnings
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# Internal imports: the PSFs are made with the exact same code as the CNN training data, so the viewer can never drift from it
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_psf_dataset import (PSFGenerator, TorchPSFGenerator, aberration_function, aberration_bounds, aberration_labels,
                                  bin_image, cuda_available, defocus_values, electron_wavelength, sets_from_shared)

##################################################################################################################################################################################################################
"""Constants"""

# Example aberration coefficients for testing, order [A1r, A1i, B2r, B2i, A2r, A2i, C3, A3r, A3i] (same as aberration_labels).
# There is no C1 here: like in the training data, C1 is set by the defocus channels (cnn_defocus).
# Keep the values inside aberration_bounds, otherwise the CNN is asked about PSFs it never saw during training.
a1r = 0;      a1i = 0;      # astigmatism            (e-8)
b2r = 0;      b2i = 0;      # coma                   (e-6)
a2r = 15e-6;  a2i = 10e-6;  # 3-fold astigmatism     (e-6)
c3  = 1e-3                  # spherical aberration   (e-3), always > 0 in the training data
a3r = 0;      a3i = 0;      # 4-fold astigmatism     (e-4)
example_aberration_vector = np.array([a1r, a1i, b2r, b2i, a2r, a2i, c3, a3r, a3i])   # order: C3 comes before A3

V = 30000                                   # Voltage [V]

# Settings of the most recent CNN (ConvolutionalNeural_Net_Iter.2/CNN_iteration_2_Through_Focus.py): it takes channels [0, 2] of the
# dataset, i.e. C1 = +5 um and -5 um, and divides the images by the pixel std of the training set (images_std in model_data.npz).
cnn_defocus = defocus_values[[0, 2]]        # C1 [m] per input channel
cnn_model_name = "CNN_iteration_2_v2_9_double_learning_rate"   # model folder whose model_data.npz holds images_std
cnn_models_folder = Path(__file__).resolve().parent.parent / "ConvolutionalNeural_Net_Iter.2" / "Trained_Models"

##################################################################################################################################################################################################################
"""Functions to generate the CNN input from aberration coefficients"""

def load_images_std(model_name=cnn_model_name):
    """images_std (float32) the CNN divided its input by during training, from Trained_Models/<model_name>/model_data.npz."""
    path = cnn_models_folder / model_name / "model_data.npz"
    if not path.exists():
        raise FileNotFoundError(f"No {path}: train that model first, pick another model_name, or pass images_std yourself")
    return np.load(path)["images_std"]

def shared_coefficients(aber):
    """[A1r, ..., A3i] (9) or a full vector [C1, A1r, ..., A3i] (10) -> the 9 coefficients the defocus channels share.
    C1 of a full vector is dropped (with a warning if it is not 0): the defocus channels set C1, exactly like in the training data."""
    aber = np.asarray(aber, dtype=float)
    if aber.shape == (len(aberration_labels) + 1,):
        if aber[0] != 0:
            warnings.warn(f"C1 = {aber[0]:.2e} m is ignored: the CNN channels use C1 = {', '.join(f'{d:+.0e}' for d in cnn_defocus)} m")
        aber = aber[1:]
    if aber.shape != (len(aberration_labels),):
        raise ValueError(f"Expected {len(aberration_labels)} coefficients {aberration_labels} (or 10 with C1 in front), got shape {aber.shape}")
    outside = (aber < aberration_bounds[:, 0]) | (aber > aberration_bounds[:, 1])
    if outside.any():
        warnings.warn(f"{[label for label, o in zip(aberration_labels, outside) if o]} outside the training bounds: the CNN never saw such PSFs")
    return aber

def make_aberration_psf(aber, images_std=None, model_name=cnn_model_name, defocus=cnn_defocus, device="auto",
                        voltage=V, alpha=20e-3, pixel_size=150e-12, n=3072, crop=1152, out_size=128):
    """Aberration coefficients [m] -> the exact input of the most recent CNN: a float32 tensor of shape (1, len(defocus), out_size, out_size).
    aber = [A1r, ..., A3i] (9 values) or [C1, A1r, ..., A3i] (10 values, C1 is ignored).
    Same pipeline as the training data (generate_psf_dataset.generate_training_dataset with its default settings) and the CNN script:
      1. one aberration vector per defocus channel, C1 = +5 um and -5 um, all other coefficients shared
      2. intensity PSF on an n x n grid, cropped to the central 2*crop pixels, normalized so the full PSF sums to 1
      3. binned (summed) to out_size x out_size, float32
      4. divided by images_std, the pixel std of the training set (loaded from the model's model_data.npz when images_std is None)
    device = "auto" (GPU if PyTorch sees one, else CPU), "cuda" or "cpu", same choice the dataset generator makes."""
    shared = shared_coefficients(aber)
    vectors = sets_from_shared(shared[None, :], defocus)[0]                  # (C, 10): one full vector per channel
    if images_std is None:
        images_std = load_images_std(model_name)

    if device == "auto":
        device = "cuda" if cuda_available() else "cpu"
    if device != "cpu":
        binned, energy = TorchPSFGenerator(voltage, alpha, pixel_size, n, crop, device)(vectors, out_size)
    else:
        generator = PSFGenerator(voltage, alpha, pixel_size, n, crop)
        psis = [generator(vector) for vector in vectors]
        binned = np.stack([bin_image(p, out_size) for p in psis]).astype(np.float32)
        energy = np.array([p.sum() for p in psis])
    if energy.min() < 0.99:
        warnings.warn(f"{100 * (1 - energy.min()):.1f}% of a PSF falls outside the crop window, the CNN input is clipped")

    images = binned[None] / np.float32(images_std)                        # (1, C, H, W) float32, standardized like in the CNN script
    return torch.from_numpy(images).to(torch.float32)

def make_aberration_psf_nostd(aber, defocus=cnn_defocus, device="auto",
                        voltage=V, alpha=20e-3, pixel_size=150e-12, n=3072, crop=1152, out_size=128):
    """Aberration coefficients [m] -> the exact input of the most recent CNN: a float32 tensor of shape (1, len(defocus), out_size, out_size).
    aber = [A1r, ..., A3i] (9 values) or [C1, A1r, ..., A3i] (10 values, C1 is ignored).
    Same pipeline as the training data (generate_psf_dataset.generate_training_dataset with its default settings) and the CNN script:
      1. one aberration vector per defocus channel, C1 = +5 um and -5 um, all other coefficients shared
      2. intensity PSF on an n x n grid, cropped to the central 2*crop pixels, normalized so the full PSF sums to 1
      3. binned (summed) to out_size x out_size, float32
      4. This version does not standarise the output, makes it easier for CNN code implementation :D
    device = "auto" (GPU if PyTorch sees one, else CPU), "cuda" or "cpu", same choice the dataset generator makes."""
    shared = shared_coefficients(aber)
    vectors = sets_from_shared(shared[None, :], defocus)[0]                  # (C, 10): one full vector per channel

    if device == "auto":
        device = "cuda" if cuda_available() else "cpu"
    if device != "cpu":
        binned, energy = TorchPSFGenerator(voltage, alpha, pixel_size, n, crop, device)(vectors, out_size)
    else:
        generator = PSFGenerator(voltage, alpha, pixel_size, n, crop)
        psis = [generator(vector) for vector in vectors]
        binned = np.stack([bin_image(p, out_size) for p in psis]).astype(np.float32)
        energy = np.array([p.sum() for p in psis])
    if energy.min() < 0.99:
        warnings.warn(f"{100 * (1 - energy.min()):.1f}% of a PSF falls outside the crop window, the CNN input is clipped")

    images = binned[None]                        # (1, C, H, W) float32, standardized like in the CNN script
    return torch.from_numpy(images).to(torch.float32)  #So then to the right outout format, in this case with no standarisation


def pupil_phases(aber, defocus=cnn_defocus, voltage=V, alpha=20e-3, n_points=401):
    """Pupil phase 2pi/lambda * phi [rad] per defocus channel on a square grid just around the aperture (NaN outside), for plotting only."""
    wavelength = electron_wavelength(voltage)
    theta = np.linspace(-1.1 * alpha, 1.1 * alpha, n_points)
    TX, TY = np.meshgrid(theta, theta)
    inside = np.hypot(TX, TY) <= alpha
    vectors = sets_from_shared(shared_coefficients(aber)[None, :], defocus)[0]
    phases = [np.where(inside, 2 * np.pi / wavelength * aberration_function(TX, TY, alpha, v), np.nan) for v in vectors]
    return theta, phases

##################################################################################################################################################################################################################
"""Plotting"""

def plot_cnn_input(cnn_input, aber, defocus=cnn_defocus, alpha=20e-3, pixel_size=150e-12, crop=1152):
    """One row per CNN input channel: pupil phase, CNN input (linear) and CNN input (log scale)."""
    images = cnn_input[0].detach().cpu().numpy()
    half = images.shape[-1] * pixel_size * (2 * crop // images.shape[-1]) / 2 * 1e9      # half width of the image [nm]
    psf_extent = [-half, half, -half, half]
    theta, phases = pupil_phases(aber, defocus, alpha=alpha)
    pupil_extent = [theta[0] * 1e3, theta[-1] * 1e3, theta[0] * 1e3, theta[-1] * 1e3]

    fig, axes = plt.subplots(len(images), 3, figsize=(17, 5 * len(images)), squeeze=False)
    for row, (image, phase, c1) in zip(axes, zip(images, phases, defocus)):
        ax1, ax2, ax3 = row
        im1 = ax1.imshow(np.angle(np.exp(1j * phase)), origin="lower", extent=pupil_extent, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax1.add_patch(plt.Circle((0, 0), alpha * 1e3, fill=False, color="k", lw=1))
        ax1.set_title(f"Pupil phase  2π/λ·φ(x', y'),  C1 = {c1 * 1e6:+.0f} µm")
        ax1.set_xlabel("θx [mrad]"); ax1.set_ylabel("θy [mrad]")
        cb1 = fig.colorbar(im1, ax=ax1, shrink=0.85, label="Phase [rad]")
        cb1.set_ticks([-np.pi, 0, np.pi]); cb1.set_ticklabels(["−π", "0", "π"])

        im2 = ax2.imshow(image, origin="lower", extent=psf_extent, cmap="inferno")
        ax2.set_title(f"CNN input, C1 = {c1 * 1e6:+.0f} µm")
        ax2.set_xlabel("x [nm]"); ax2.set_ylabel("y [nm]")
        fig.colorbar(im2, ax=ax2, shrink=0.85, label="Ψ / images_std")

        im3 = ax3.imshow(image, origin="lower", extent=psf_extent, cmap="inferno", norm=LogNorm(vmin=image.max() * 1e-4, vmax=image.max()))
        ax3.set_title(f"CNN input, C1 = {c1 * 1e6:+.0f} µm, log scale")
        ax3.set_xlabel("x [nm]"); ax3.set_ylabel("y [nm]")
        fig.colorbar(im3, ax=ax3, shrink=0.85, label="Ψ / images_std")

        for ax in row:
            ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()

##############################################################################################
""""Run this file"""

def main():
    """This functions runs everything from this file"""
    cnn_input = make_aberration_psf_nostd([example_aberration_vector])
    print(np.shape(cnn_input))
    print(cnn_input)
if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
