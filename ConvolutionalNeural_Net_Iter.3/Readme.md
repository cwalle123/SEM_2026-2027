# Convolutional Neural Network Iteration 3

Same network and settings as iteration 2 (copied from it), adapted to datasets made with `generate_psf_dataset_NEW.py`:

- **Only 2 channels in the data:** overfocus (C1 = +5 µm) and underfocus (C1 = −5 µm). The focused PSF is no longer generated, so the images are used as they are (no `[:, [0, 2]]` selection).
- **Simple label normalisation:** labels = c / (high − low), so 0 m stays label 0. Predictions are converted back with `labels * (high - low)`.
- **Data folder:** the datasets are loaded from `Data/` in the repository root (where `generate_psf_dataset_NEW.py` saves them). Datasets made with the old generator are refused, because they have 3 channels and the old [−1, 1] labels.
- **Single-vector analysis** makes its PSFs with `generate_psf_dataset_NEW.py` directly (no `psf_viewer`).

## Version V1

Version V1 plays around with some hyperparameters, and changing the kernel sizes wrt to the previous CNN iteration to increase generalisation performance and also simplified the structure of the CNN (removing one conv. layer to also improove generalisation performance).

The CNN structure is defined as follows 

```ruby
image_size=128 #square images considered (imagesize x imageize)
n_in_channels=2 #Images taken as input onto the CNN


#Below the channels per each parallel pipeline (depthwise convolution)---

#Layer 1
n_indiv_channels_c1=32 ; n_indiv_channels_c2=32 
kernel_c1= 4           ; kernel_c2= 8
max_pool_1=2
#Layer2
n_indiv_channels_c3=32 ; n_indiv_channels_c4=32
kernel_c3= 4           ;    kernel_c4= 4 
max_pool_2=2
#Dense layers
p_dropout=0.2
n_neurons=1000
```

With the last reported values for training and validation shown below

```
Current epoch: 3700
Training_loss=0.0002636689750943333
Validation_loss=0.0005967800971120596
```
## Some results and comparaison with iter.2's V3

### Iter.3
<img width="1917" height="937" alt="image" src="https://github.com/user-attachments/assets/624deff8-7cc9-4543-a1fa-712ab3e415bc" />
<img width="1895" height="932" alt="image" src="https://github.com/user-attachments/assets/727bf6e2-5ff2-4886-9faa-534c6d203af3" />

### Iter.2
<img width="1487" height="977" alt="image" src="https://github.com/user-attachments/assets/83e23832-e1b9-4ae5-8caf-a8174de23741" />
<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/56bdb070-4374-4aee-a2e5-f4e0f7d7e345" />


### Validation overview 

<img width="1881" height="927" alt="image" src="https://github.com/user-attachments/assets/63a68d9c-913b-440e-a0cf-e64550bef3e9" />

### Version V2

Increased the number of filters on the first layer to make curvature detection more precise and thus reduce A2 and A3 errors. The learning rate was kept constant. The structure looks as follows:



