# PSF from Image

Simulates wave-optical point spread functions (PSFs) of an SEM electron probe from a set of lens aberration coefficients, and generates a labelled dataset for training a CNN to predict the aberrations from PSF images.

## Files

| File                      | What it does                                                                                             |
|---------------------------|----------------------------------------------------------------------------------------------------------|
| `WavePSF0.py`             | The old PSF script from Noah, kept as the reference for the wave-optical PSF calculation                 |
| `psf_viewer.py`           | Computes a single PSF from one aberration vector and plots the PSF intensity to see what it looks like.  |
| `generate_psf_dataset.py` | Makes training data: random aberration vectors with 3 set C1 + their images → `Data/psf_<k>coef_….npz`.  |
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

By default this generates 1000 samples of the 3-coefficient dataset. In `main()`, change `n_samples` for more or fewer samples and `n_coefficients` (3, 7 or 9) for the dataset type.

### Dataset types

`n_coefficients` sets how many coefficients are random. Every coefficient that is not in the chosen set is exactly 0. C1 is not counted: it is always used, fixed per channel (see below). So the vectors have 4, 8 or 10 coefficients in use, but they are always stored as the full 10-number vector with zeros in the unused slots.

| `n_coefficients` | Random coefficients                   | Filename (for `n_samples = 1000`)  | Why this set                                                                                                               |
|------------------|---------------------------------------|------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| 3                | A1r, A1i, C3                          | `psf_3coef_A1-C3_n1000.npz`        | Classic uncorrected-SEM probe: C3 is unavoidable for round lenses, A1 is corrected by the operator every session [3], [4]  |
| 7                | A1r, A1i, B2r, B2i, A2r, A2i, C3      | `psf_7coef_A1-B2-A2-C3_n1000.npz`  | Exactly the aberrations (with C1) retrieved from SEM through-focus series by Uno et al. [1]: everything up to second order plus Cs |
| 9                | All: A1r … A3i                        | `psf_9coef_A1-B2-A2-C3-A3_n1000.npz` | Everything up to third order in this model                                                                               |

Each type gets its own file, cache and preview folder, so the types don't overwrite each other. You can also pass your own `filename`.

### What one sample is

Each sample is an **aberration set**: one random draw of the active coefficients, imaged at three fixed defocus values, **C1 = +5 µm, 0, −5 µm**. This gives a 3-channel image (one channel per defocus). The network has to predict the 3, 7 or 9 shared coefficients. C1 is not a target because it is fixed per channel.

The coefficients are drawn uniformly from `aberration_bounds` at the top of the file. The defaults are chosen for an uncorrected SEM objective lens at 30 kV and α = 20 mrad:

| Coefficient | Range                        | Basis                                                                                                            |
|-------------|------------------------------|------------------------------------------------------------------------------------------------------------------|
| A1 (r, i)   | ±1 µm                        | ~10–100 nm after stigmation [1]; much larger before stigmation, since it drifts and must be re-corrected [3]     |
| B2 (r, i)   | ±10 µm                       | ~0.1 µm when aligned [1]; beam misalignment gives B2 ≈ Cs · tilt ≈ 2 mm × 5 mrad ≈ 10 µm                          |
| A2 (r, i)   | ±20 µm                       | ~1–2 µm measured on a real SEM; 20 µm is the upper end tested in [5]                                             |
| C3          | 0.5 – 5 mm (always positive) | 1.9 mm for a conical objective at WD = 3 mm [2]; Cs grows roughly as WD³. Always > 0 for round lenses (Scherzer)  |
| A3 (r, i)   | ±200 µm                      | Third-order parasitic, rarely measured on SEMs [4]; ~10× the π/4 tolerance                                       |

Out-lens SEMs at WD ≈ 10 mm have Cs of tens of mm, but those are used with α ≈ 5 mrad, not 20 mrad, so they are not in the range.

The 5 µm defocus alone already spreads the beam over ±100 nm of the ±173 nm crop window. The ranges were checked numerically: even with every coefficient at a worst-case extreme, more than 99.9% of each PSF stays inside the crop and the pupil phase is not undersampled.

**References**

1. S. Uno, K. Honda, N. Nakamura, M. Matsuya, J. Zach, "Aberration correction and its automatic control in scanning electron microscopes", *Optik* 116 (2005) 438–448.
2. H. Kazumori et al., "Field Emission SEM with a Newly Developed FEGUN and Conical Strongly Excited Objective Lens", *Microsc. Microanal.* 6 (S2) (2000) 764–765.
3. M. Rudnaya, *Automated focusing and astigmatism correction in electron microscopy*, PhD thesis, TU Eindhoven (2011).
4. O. L. Krivanek, N. Dellby, A. R. Lupini, "Towards sub-Å electron beams", *Ultramicroscopy* 78 (1999) 1–11.
5. R. Dortland, *A benchmark study on SEM aberration retrieval*, BSc thesis, TU Delft (2024).

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

Progress is cached in `Data/` (`<dataset name>_partial_images.npy` and `<dataset name>_partial_progress.json`) every 10 sets. If a run stops, run the script again with the **same settings** and it continues where it left off. The cache files are deleted once the final `.npz` has been saved. If you change the settings, delete the cache files first or use a different `filename`.

### Output

Each dataset file (for example `Data/psf_3coef_A1-C3_n1000.npz`) contains, with K = 3, 7 or 9:

| Key                                                   | Shape                    | Contents                                                              |
|-------------------------------------------------------|--------------------------|-----------------------------------------------------------------------|
| `images`                                              | (N, 3, 128, 128) float32 | The CNN input: one binned PSF per defocus channel                     |
| `labels`                                              | (N, K) float32           | The CNN target: the active coefficients scaled to [−1, 1]             |
| `aberration_vectors`                                  | (N, 3, 10)               | The full vectors for each channel [m], unused coefficients are 0      |
| `defocus_values`                                      | (3,)                     | C1 of each channel [m]                                                |
| `bounds`, `label_names`                               | (K, 2), (K,)             | Bounds and names of the active coefficients, to convert labels back to metres |
| `vector_labels`                                       | (10,)                    | Names of the 10 entries of the full vector                            |
| `n_coefficients`, `active_mask`                       | scalar, (9,)             | Dataset type, and which of A1r … A3i were random                      |
| `train_idx`, `val_idx`, `test_idx`                    |                          | The fixed dataset split                                               |
| `voltage`, `alpha`, `pixel_size`, `binned_pixel_size` |                          | The microscope settings the images were made with                     |

The labels are scaled to [−1, 1] because the raw coefficients differ by several orders of magnitude. Without scaling, C3 (~10⁻³ m) would dominate the loss over A1 (~10⁻⁷ m).

`Data/<dataset name>_previews/preview_00000.png` … `preview_00002.png` show the first 3 samples, with the three defocus channels side by side. Each panel is scaled to its own maximum, so the previews are for a quick look only.

### Loading the dataset

```python
import numpy as np

data = np.load("Data/psf_3coef_A1-C3_n1000.npz")
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
