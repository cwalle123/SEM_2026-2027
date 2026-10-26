"""This file serves a generated PSF dataset to the CNN in PyTorch. It is the only file (apart from
   the CNN itself) that needs PyTorch."""

#################################################################################################################################################################
"""Imports"""

# External imports
import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Internal imports
from constants import DATA_FOLDER, DATASET_NAME, COEFF_NAMES
from Data_PSF_Generator import load_dataset, half_ranges

#################################################################################################################################################################
"""Info"""

# PyTorch trains on a Dataset object: something that can say how many samples it has (__len__)
# and hand out sample number i (__getitem__). A DataLoader then puts samples into shuffled batches.
# That is why this is a class, where everything else in this folder is plain functions.
#
# One sample is (x, y):
#   x : the CNN input, the through-focus stack, shape (planes, 128, 128). By default the whole
#       stack is divided by its highest pixel, so values are between 0 and 1 while the relative
#       brightness of the planes (which carries information) is kept.
#   y : the label, the 10 coefficients divided by half their range, so roughly between -1 and 1.
#       to_coeffs() turns network outputs back into metres.
#
# Shot noise: with dose set, Poisson noise is added every time a sample is read, so every epoch
# sees different noise (the network can't memorise it). dose is the number of counts at the
# brightest pixel; a tuple (low, high) draws a random dose per sample.
#
# Usage:
#   python Data_PSF_Torch_Dataset.py                  # loads Data/psf_v1 and prints a quick check
#
#   from Data_PSF_Torch_Dataset import PSFDataset
#   train_set = PSFDataset('Data/psf_v1', indices=train_indices, dose=(50, 1000))

#################################################################################################################################################################
"""Dataset"""

class PSFDataset(Dataset):

    def __init__(self, folder, normalise=True, dose=None, indices=None):
        """
        folder    : dataset folder, e.g. Data/psf_v1
        normalise : True divides every stack by its highest pixel; False gives the fraction of the
                    beam current per pixel
        dose      : None (no noise), counts at the brightest pixel, or (low, high) for a random dose
        indices   : which samples to use (e.g. for a train/validation split); None uses all
        """
        dataset = load_dataset(folder)
        self.images = dataset['images']
        self.scales = dataset['scales']
        self.coeffs = dataset['coeffs']
        self.meta = dataset['meta']
        self.half_range = half_ranges(self.meta)
        self.normalise = normalise
        self.dose = dose
        if indices is None:
            self.indices = np.arange(len(self.coeffs))
        else:
            self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        # Undo the per-image scaling of the stored float16 images -> fraction of the current per pixel
        x = torch.from_numpy(self.images[i].astype(np.float32) * self.scales[i][:, None, None])
        if self.normalise:
            x = x / x.max()

        if self.dose is not None:
            if isinstance(self.dose, tuple):
                # torch's random generator is used (not numpy's), because the DataLoader gives every
                # worker process its own torch seed, so workers don't repeat the same noise
                dose = self.dose[0] + (self.dose[1] - self.dose[0]) * torch.rand(()).item()
            else:
                dose = self.dose
            peak = x.max()
            x = torch.poisson(x / peak * dose) / dose * peak

        y = torch.from_numpy((self.coeffs[i] / self.half_range).astype(np.float32))
        return x, y

    def to_coeffs(self, y):
        """
        Turn labels or network outputs (scaled to roughly -1..1) back into coefficients.

        y : array (..., 10)

        Returns array (..., 10) in m.
        """
        return np.asarray(y) * self.half_range

#################################################################################################################################################################
"""Main"""

def main():
    """
    Quick check that a dataset loads the way the CNN will see it.
    """
    p = argparse.ArgumentParser(description='Check a PSF dataset through the PyTorch Dataset')
    p.add_argument('--name', type=str, default=DATASET_NAME, help=f'folder name in Data/ (default {DATASET_NAME})')
    args = p.parse_args()

    dataset = PSFDataset(os.path.join(DATA_FOLDER, args.name), dose=(50, 1000))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    x, y = next(iter(loader))

    print(f'{len(dataset)} samples')
    print(f'one batch: x {tuple(x.shape)} (min {x.min():.3f}, max {x.max():.3f}),  y {tuple(y.shape)}')
    print('first sample, coefficients:')
    for name, value in zip(COEFF_NAMES, dataset.to_coeffs(y[0].numpy())):
        print(f'  {name:4s} = {value: .4e} m')

if __name__ == '__main__':
    main()
