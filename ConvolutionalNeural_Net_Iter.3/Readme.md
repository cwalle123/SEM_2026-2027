# Convolutional Neural Network Iteration 3

Same network and settings as iteration 2 (copied from it), adapted to datasets made with `generate_psf_dataset_NEW.py`:

- **Only 2 channels in the data:** overfocus (C1 = +5 µm) and underfocus (C1 = −5 µm). The focused PSF is no longer generated, so the images are used as they are (no `[:, [0, 2]]` selection).
- **Simple label normalisation:** labels = c / (high − low), so 0 m stays label 0. Predictions are converted back with `labels * (high - low)`.
- **Data folder:** the datasets are loaded from `Data/` in the repository root (where `generate_psf_dataset_NEW.py` saves them). Datasets made with the old generator are refused, because they have 3 channels and the old [−1, 1] labels.
- **Single-vector analysis** makes its PSFs with `generate_psf_dataset_NEW.py` directly (no `psf_viewer`).

## Differences regarding the V2 in regaring CNN structure:

### Version V1

Version V1 plays around with some hyperparameters, and changing the kernel sizes wrt to the previous CNN iteration to increase generalisation performance and also simplified the structure of the CNN (removing one conv. layer to also improove generalisation performance)


### Version V2