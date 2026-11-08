# PSF from Image

Simulates wave-optical point spread functions (PSFs) of an SEM electron probe from a set of lens aberration coefficients, and generates a labelled dataset for training a CNN to predict the aberrations from PSF images.

## Files

| File                      | What it does                                                                                             |
|---------------------------|----------------------------------------------------------------------------------------------------------|
| `WavePSF0.py`             | The old PSF script from Noah, kept as the reference for the wave-optical PSF calculation                 |
| `psf_viewer.py`           | Computes a single PSF from one aberration vector and plots the PSF intensity to see what it looks like.  |
| `generate_psf_dataset.py` | Makes training data: random aberration vectors with 3 set C1 + their images → `Data/psf_dataset.npz`.    |
| `Data/`                   | Output folder for the dataset and the preview images.                                                    |

## How the PSF is calculated

1. **Wavelength:** the relativistic electron wavelength is computed from the accelerating voltage (30 kV by default).
2. **Wave aberration:** the aberration function φ(θx, θy) is evaluated over the beam angles inside the aperture (half opening angle α = 20 mrad by default). The coma and spherical terms are balanced against lower orders, as in Zernike polynomials.
3. **Pupil function:** P = A · exp(i · 2π/λ · φ), where A = 1 inside the aperture and 0 outside.
4. **PSF:** the complex probe is the Fourier transform of the pupil. The intensity PSF is Ψ = |FT{P}|², normalised to sum to 1 so it can be used as a convolution kernel.

The angular step of the pupil grid is λ / (n · pixel_size), so the FFT lands directly on image pixels of size `pixel_size`. Both scripts warn if the pupil phase changes by more than π per pixel (undersampling). If that happens, increase `n`.

### Aberration vector

Every aberration is described by a 10-number vector in metres, in this order (C3 comes before A3):

```
[C1, A1r, A1i, B2r, B2i, A2r, A2i, C3, A3r, A3i]
```

| Symbol | Aberration                | Components          |
|--------|---------------------------|---------------------|
| C1     | Defocus                   | 1                   |
| A1     | Two-fold astigmatism      | 2 (real, imaginary) |
| B2     | Axial coma                | 2                   |
| A2     | Three-fold astigmatism    | 2                   |
| C3     | Spherical aberration (Cs) | 1                   |
| A3     | Four-fold astigmatism     | 2                   |

Round aberrations (C1, C3) only have a size. The others also have a direction, so they have an x (real) and a y (imaginary) component.

### Differences from `WavePSF0.py`

- The four-fold astigmatism term has a 1/4 prefactor (the same 1/n pattern as A1 and A2). `WavePSF0.py` has no prefactor on this term.
- There is no Gaussian blur for the finite source size (`d50geo` in `WavePSF0.py`) yet, see [To do](#to-do).
- The wavelength is computed from physical constants instead of the 1.226 nm / √U approximation. The results are practically the same.

## Viewing a single PSF

```
python psf_viewer.py
```

This plots the PSF for `example_abberation_vector` at the top of the file (by default 0.5 µm defocus plus 10 µm three-fold astigmatism). Change the values there to try other aberrations. In `main()` you can also change the voltage, α, pixel size, grid size `n`, and the plotted window `crop`.

## Generating the training dataset

```
python generate_psf_dataset.py
```

By default this generates 1000 samples. Change `n_samples` in `main()` for more or fewer.

### What one sample is

Each sample is an **aberration set**: one random draw of the 9 coefficients A1r … A3i, imaged at three fixed defocus values, **C1 = +5 µm, 0, −5 µm**. This gives a 3-channel image (one channel per defocus). The network has to predict the 9 shared coefficients. C1 is not a target because it is fixed per channel.

The coefficients are drawn uniformly from `aberration_bounds` at the top of the file. The defaults are chosen for an uncorrected SEM objective lens at 30 kV and α = 20 mrad:

| Coefficient | Range                        |
|-------------|------------------------------|
| A1 (r, i)   | ±500 nm                      |
| B2 (r, i)   | ±5 µm                        |
| A2 (r, i)   | ±20 µm                       |
| C3          | 0.5 – 2 mm (always positive) |
| A3 (r, i)   | ±200 µm                      |

### Default settings

| Setting                         | Default              | Meaning                                                                     |
|---------------------------------|----------------------|-----------------------------------------------------------------------------|
| `voltage`                       | 30 kV                | Accelerating voltage                                                        |
| `alpha`                         | 20 mrad              | Half opening angle                                                          |
| `pixel_size`                    | 150 pm               | PSF pixel size before binning                                               |
| `n`                             | 3072                 | FFT grid size                                                               |
| `crop`                          | 1152                 | Half width of the window around the centre, in pixels (2304 px ≈ 346 nm)    |
| `out_size`                      | 128                  | Final image size after binning (18 × 18 pixels summed → 2.7 nm pixels)      |
| `val_fraction`, `test_fraction` | 0.1, 0.1             | Fixed 80 / 10 / 10 train / validation / test split                          |
| `seed`                          | 0                    | Makes the dataset reproducible                                              |
| `n_workers`                     | Half the CPU threads | Parallel processes. Each uses 0.5–1GB of RAM, so lower if run out of memory |

`2 * crop` must be divisible by `out_size`, and `crop` must be at most `n // 2`. The script warns if more than 1% of a PSF falls outside the crop window. If you see that warning, increase `crop` (and `n`).

### Resuming an interrupted run

Progress is cached in `Data/` (`psf_dataset_partial_images.npy` and `psf_dataset_partial_progress.json`) every 10 sets. If a run stops, run the script again with the **same settings** and it continues where it left off. The cache files are deleted once the final `.npz` has been saved. If you change the settings, delete the cache files first or use a different `filename`.

### Output

`Data/psf_dataset.npz` contains:

| Key                                                   | Shape                    | Contents                                          |
|-------------------------------------------------------|--------------------------|---------------------------------------------------|
| `images`                                              | (N, 3, 128, 128) float32 | The CNN input: one binned PSF per defocus channel |
| `labels`                                              | (N, 9) float32           | The CNN target: coefficients scaled to [−1, 1]    |
| `aberration_vectors`                                  | (N, 3, 10)               | The full vectors for each channel [m]             |
| `defocus_values`                                      | (3,)                     | C1 of each channel [m]                            |
| `bounds`, `label_names`, `vector_labels`              |                          | Needed to convert labels back to metres           |
| `train_idx`, `val_idx`, `test_idx`                    |                          | The fixed dataset split                           |
| `voltage`, `alpha`, `pixel_size`, `binned_pixel_size` |                          | The microscope settings the images were made with |

The labels are scaled to [−1, 1] because the raw coefficients differ by several orders of magnitude. Without scaling, C3 (~10⁻³ m) would dominate the loss over A1 (~10⁻⁷ m).

`Data/preview_00000.png` … `preview_00009.png` show the first 10 samples, with the three defocus channels side by side. Each panel is scaled to its own maximum, so the previews are for a quick look only.

### Loading the dataset

```python
import numpy as np

data = np.load("Data/psf_dataset.npz")
X, y = data["images"], data["labels"]

X_train, y_train = X[data["train_idx"]], y[data["train_idx"]]
X_val,   y_val   = X[data["val_idx"]],   y[data["val_idx"]]
X_test,  y_test  = X[data["test_idx"]],  y[data["test_idx"]]

# Convert network predictions (scaled to [-1, 1]) back to metres
low, high = data["bounds"][:, 0], data["bounds"][:, 1]
coefficients_m = low + (predictions + 1) / 2 * (high - low)
```

The images are normalised so that the full PSF sums to 1, which makes the pixel values very small. You will probably want to rescale each image (for example, divide by its maximum) before feeding it to the CNN.

## To do

- [ ] Add the Gaussian blur for the finite source size (`d50geo`) from `WavePSF0.py`, to `psf_viewer.py` and `generate_psf_dataset.py`.

## Requirements

- Python 3
- numpy
- scipy (used for the multi-threaded FFT in `generate_psf_dataset.py`)
- matplotlib
