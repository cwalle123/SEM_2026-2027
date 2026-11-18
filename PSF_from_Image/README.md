# PSF from Image

Simulates wave-optical point spread functions (PSFs) of an SEM electron probe from a set of lens aberration coefficients, and generates a labelled dataset for training a CNN to predict the aberrations from PSF images.

## Files

| File                      | What it does                                                                                             |
|---------------------------|----------------------------------------------------------------------------------------------------------|
| `WavePSF0.py`             | The old PSF script from Noah, kept as the reference for the wave-optical PSF calculation                 |
| `psf_viewer.py`           | Computes a single PSF from one aberration vector and plots the PSF intensity to see what it looks like.  |
| `generate_psf_dataset.py` | Makes training data: random aberration vectors with 3 set C1 + their images → `Data/psf_<k>coef_….npz`. **NEW GRID METHOD HERE:** also makes grid data on the fly for the CNN, without saving anything (see [Grid dataset](#new-grid-method-here-grid-dataset-made-on-the-fly-nothing-saved)). |
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

> **NEW GRID METHOD HERE:** `main()` now has two switches at the top:
> - `make_random`: makes and saves the random dataset described in this section (10 000 samples of the 9-coefficient type). It is `False` by default, so running the file doesn't regenerate it.
> - `check_grid`: builds the grid datasets and runs one pass over the grid as a check. It is `True` by default, and nothing is saved. See [Grid dataset](#new-grid-method-here-grid-dataset-made-on-the-fly-nothing-saved).

For the random dataset, set `make_random = True` in `main()`. Change `n_samples` for more or fewer samples, and `n_coefficients` (3, 7 or 9) for the dataset type.

The PSFs are computed on the GPU (PyTorch + CUDA) when one is available, and on the CPU with parallel worker processes otherwise. On an RTX 3070 one set of 3 PSFs takes about 0.025 s, so 10 000 samples take about 4 minutes. Use `device="cpu"` or `device="cuda"` to force one or the other.

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
| `n_workers`                     | Half the CPU threads | Parallel processes (CPU only). Each uses 0.5–1GB of RAM, so lower if run out of memory |
| `device`                        | `"auto"`             | `"auto"` = GPU if PyTorch sees one, else CPU; or force `"cuda"` / `"cpu"`   |
| `gpu_batch_sets`                | 4                    | Sets (× 3 PSFs) per GPU FFT batch, ~2.8 GB of GPU memory. Lower it on a CUDA out-of-memory error |

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

## NEW GRID METHOD HERE: Grid dataset (made on the fly, nothing saved)

The random dataset above draws coefficients at random. The grid dataset uses **every combination of fixed values** inside `aberration_bounds` instead, so you know exactly which values the network was trained on. The validation set uses only values that are **not** in the training grid.

Nothing is written to disk. Each sample is fully determined by its index (index → position on the grid → coefficients), so only the grid definition is stored. The PSFs are computed on the GPU at the moment the CNN asks for them, one batch at a time, and each batch is freed once training moves on to the next one.

### The grid

`step` sets the spacing:

- **A number** is a fraction of each coefficient's range. `0.1` gives 11 values per coefficient (low, low + 10 %, …, high), `0.25` gives 5, and `0.5` gives 3 (low, middle, high). On the [−1, 1] label scale, the step is `2 * step`.
- **A dict** gives a step in metres per coefficient, keyed by label (`"A1r"`) or by group (`"A1"` = A1r and A1i). For example `step={"A1": 0.2e-6, "C3": 0.5e-3}`.

The grid starts at the low bound and ends at the last step that still fits, which is the high bound when the range is a whole number of steps.

- **Training grid:** all grid points.
- **Validation grid:** all **midpoints** between neighbouring grid points, which gives one fewer value per coefficient. None of them is in the training grid, so validation always tests values the network has never seen. With `step=0.5`, the training labels are {−1, 0, 1} and the validation labels are {−0.5, 0.5}.

The number of sets is (values per coefficient)^K, which grows very fast with the number of coefficients K:

| `n_coefficients` | `step` | Training sets   | Validation sets | One pass over the training grid (RTX 3070, 2 channels) |
|------------------|--------|-----------------|-----------------|--------------------------------------------------------|
| 3                | 0.1    | 11³ = 1331      | 10³ = 1000      | ~22 s                                                  |
| 3                | 0.05   | 21³ = 9261      | 20³ = 8000      | ~2.5 min                                               |
| 9                | 0.5    | 3⁹ = 19 683     | 2⁹ = 512        | ~5.5 min                                               |
| 9                | 0.25   | 5⁹ ≈ 2 million  | 4⁹ = 262 144    | ~9 hours                                               |

With 7 or 9 coefficients, the grid can only be very coarse. That is why the random dataset is the better choice there. The grid works best with 3 coefficients.

### Usage in the CNN

To check the grid without the CNN, run `python generate_psf_dataset.py` with `check_grid = True` in `main()`. It builds the step-0.1, 3-coefficient grids and runs one pass over the training grid in batches of 500. It prints each batch's shape and saves nothing.

To train on the grid, use the datasets like this in the CNN script:

```python
import sys
sys.path.append(f"{Base_Directory}/PSF_from_Image")          # so the CNN can import the generator
from generate_psf_dataset import grid_datasets, defocus_values

# Training grid + validation grid (midpoints between the training points)
train_set, val_set = grid_datasets(step=0.1, n_coefficients=3,
                                   defocus=defocus_values[[0, 2]])   # channels: +5 µm and -5 µm

for epoch in range(n_epochs):
    for x_batch, y_batch in train_set.batches(batch_size=500):    # (500, 2, 128, 128), (500, 3)
        optimizer.zero_grad()
        loss = lossfn(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()

    with torch.no_grad():                                         # validation, also batch by batch
        val_loss = sum(lossfn(model(x), y).item() * len(y) for x, y in val_set.batches(500)) / len(val_set)
```

The CNN in `ConvolutionalNeural_Net_Iter.2` does not use this yet. It still loads a saved `.npz` file.

### Options and helpers

| What                                                    | Meaning                                                                                                                                      |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `grid_datasets(step, n_coefficients, **options)`        | Returns `(train_set, val_set)`. The options are the same as for `GridPSFDataset`                                                            |
| `GridPSFDataset(step, n_coefficients, midpoints=False)` | One grid. `midpoints=True` gives the validation grid                                                                                        |
| `defocus=defocus_values[[0, 2]]`                        | Which C1 values become the image channels. The default is all 3 (+5, 0, −5 µm); fewer channels means less computing                         |
| `.batches(batch_size=500, shuffle=True, prefetch=True)` | One epoch: every grid point exactly once, in a new random order every call                                                                  |
| `prefetch=True`                                         | Makes the next batch in a background thread while the CNN trains on the current one: at most 2 batches in memory (~65 MB each for 500 × 2 channels) |
| `prefetch=False`                                        | Exactly 1 batch in memory, but training waits while each batch is made (~3.3 s per 500 sets × 2 channels)                                   |
| `dataset[idx]`                                          | Specific samples (int, list/array or slice), made right then and not kept                                                                   |
| `len(dataset)`, `.n_batches(batch_size)`                | Number of sets, and number of batches per epoch                                                                                             |
| `.labels(idx)`, `.coefficients(idx)`, `.aberration_vectors(idx)` | Labels in [−1, 1], coefficients in metres, and the full 10-number vectors, all without making the images                           |
| `.bounds`, `.label_names`                               | Same as in the `.npz` files, to convert predicted labels back to metres                                                                     |
| `.image_std()`                                          | Standard deviation of the pixel values from a random sample of the grid, to standardise the images like the CNN does now                    |
| `device`, `out_device`                                  | Where the PSFs are computed (`"auto"` = GPU if available) and where the returned tensors go (default `"cpu"`)                                |

The images and labels come back as float32 PyTorch tensors. They are scaled the same way as the `images` and `labels` in the saved `.npz` files.

## To do

- [ ] Add the Gaussian blur for the finite source size (`d50geo`) from `WavePSF0.py`, to `psf_viewer.py` and `generate_psf_dataset.py`.

## Requirements

- Python 3
- numpy
- scipy (used for the multi-threaded FFT in `generate_psf_dataset.py`)
- matplotlib
- PyTorch with CUDA, for the GPU. **NEW GRID METHOD HERE:** the grid dataset always needs PyTorch, and runs on the CPU when there is no GPU. The random dataset works without PyTorch, but then runs on the CPU only.
