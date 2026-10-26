# Wave PSF → CNN aberration retrieval

Generates wave-optical SEM probe shapes (PSFs) with known aberration coefficients, as training data
for a PyTorch CNN that predicts the coefficients from the images.

The physics is the one in `WavePSF0.py` (Zernike–Nijboer phase in the aperture → FFT → |ψ|² →
Gaussian source blur d50_geo). `Model_PSF_Wave_Optics.py` is a batched version of it, checked
against the original to ~1e-7 of the peak.

## Files

| file                                     | what it does                                                                                                                      |
|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `constants.py`                           | **every setting**: machine (voltage, θ, d50_geo), image sampling, through-focus planes, coefficient ranges, dataset size, folders |
| `Model_PSF_Wave_Optics.py`                 | the physics: electron wavelength, aberration function, PSF stack                                                                |
| `Data_PSF_Generator.py`                    | generates a dataset into `Data/<name>/` and loads it back                                                                       |
| `Data_PSF_Torch_Dataset.py`                | `PSFDataset`: serves a dataset to the CNN (the only PyTorch file so far)                                                        |
| `Validation_PSF_Check_Against_Original.py` | checks the PSFs against `WavePSF0.py`                                                                                           |
| `Validation_PSF_Previews.py`               | figures 01–05 in `Figures/`                                                                                                     |
| `WavePSF0.py`                              | original PSF code (reference, unchanged)                                                                                        |

## How to run

```
python Validation_PSF_Check_Against_Original.py   # same PSFs as WavePSF0.py?  -> PASS
python Data_PSF_Generator.py                      # 10 000 samples -> Data/psf_v1  (~6 min, ~1 GB)
python Data_PSF_Torch_Dataset.py                  # loads it the way the CNN will
python Validation_PSF_Previews.py                 # figures -> Figures/
```

To change anything (ranges, image size, defocus planes, machine settings), edit `constants.py`
and generate a new dataset, e.g. `python Data_PSF_Generator.py --name psf_v2`.

## What one sample is

* **Input:** a through-focus stack of 3 images, 128 × 128 px of 0.3 nm (window 38.4 nm), at
  C1 − 0.2 µm, C1 and C1 + 0.2 µm. These are the 3 input channels of the CNN.
* **Label:** the 10 coefficients `C1, A1r, A1i, B2r, B2i, A2r, A2i, C3, A3r, A3i` (m), each drawn
  uniformly from `COEFF_RANGES`.

**Why a stack and not one PSF:** flipping the sign of all even aberrations (C1, A1, C3, A3) gives
exactly the same single PSF (`Figures/03_sign_ambiguity.png`), so the sign can't be learned from one
image. The defocused planes break this ambiguity, which is also why the report uses under- and
overfocus.

Machine settings are fixed for now (30 kV, θ = 15 mrad, d50_geo = 1 nm) but stored per sample;
widen `MACHINE_RANGES` to train for other settings. The pixel size is fixed in nm, so datasets
made with a different θ or voltage still line up pixel for pixel.

## Using it for training

```python
import torch
from torch.utils.data import DataLoader
from Data_PSF_Torch_Dataset import PSFDataset

order = torch.randperm(10000, generator=torch.Generator().manual_seed(0)).numpy()
train_set = PSFDataset('Data/psf_v1', indices=order[:9000], dose=(50, 1000))  # new shot noise every epoch
val_set = PSFDataset('Data/psf_v1', indices=order[9000:])
x, y = train_set[0]                 # x: (3, 128, 128), stack scaled to peak 1;  y: (10,) roughly -1..1
coeffs_m = train_set.to_coeffs(y)   # back to metres
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
```

## Next

1. Supervised: CNN on the stack → 10 coefficients, loss on the scaled labels.
2. Later, for real SEM images (no labels): fine-tune with a physics loss, i.e. regenerate the PSF
   from the predicted coefficients and compare it with the measured one. That needs a PyTorch copy
   of `psf_field` so gradients can flow through it.

Requires: numpy, scipy, matplotlib, torch.
