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


#Below the channels per layer

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

#Dataset generation related

n_batch=400 #Determines minibatch size (number of samples the CNN runs over in one epoch), neeeds to be smaller than the traininig set.
training_set_fraction=0.8
validation_set_fraction=0.2
learning_rate=0.0001

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

This ersion uses again the samse strucure as the V3 from iteration 2, to check whether the different structure of the CNN improoved generalisation performance or whether it was related to the normalisation changes as well as the larger dataset.

Training is as fast as in the iter.2 while it tends to behave almost equally to the iter.2. A minimum validation lower than before is observed with the new dataset (both size and new normalisation), reaching validation losses of around 0.008 as minimum during training. Further iterations reach also an asymptotic value for the training loss, at around 0.0007, shich is above the previous training errors. Therefore, it is thought that the previous iterartion's structure is too simple to grasp detailed features of the PSFs. Even if the lr is reduced, same behaviour is obseved

The structure used is then:

```ruby
image_size=128 #square images considered (imagesize x imageize)
n_in_channels=2 #Images taken as input onto the CNN


#Layer 1
n_indiv_channels_c1=16 ; n_indiv_channels_c2=32 
kernel_c1= 9           ; kernel_c2= 5
max_pool_1=2
#Layer2
n_indiv_channels_c3=32 ; n_indiv_channels_c4=16
kernel_c3= 3           ;    kernel_c4= 3
max_pool_2=2
#Dense layers
p_dropout=0.2
n_neurons=1000

#Dataset generation related

n_batch=400 #Determines minibatch size (number of samples the CNN runs over in one epoch), neeeds to be smaller than the traininig set.
training_set_fraction=0.8
validation_set_fraction=0.2
learning_rate=0.0001
```

### Version V3

In this version inishgts gained from the V2 were used. Therefore, increasing model complexity will tend to decrease both validation and training errors by a lager significant factor than could training set conditioning as well as size. 

The structure used will increase the number of channles and kernel size for the first layer (trying to reserve it for filters used primary fueature detection, such as edge and curvature detection) to increase the resolution of the angles and features different filters (or kernels) could distinguish. This is mainly based on the results of the V1, where the features which require precise measurements of curvature and slope of the PSF contour are required.

kenel size is also increased slightly to allow for better resolution filters (which is required if we increase channels for more precise feature extractions). Note for this try, I'm trying to force CNN to use furst layer for edge detection, so if 8x8 kervel only about 8-10 different angled may be distinguished and thus only 8-10 filters (and then channels would be required)

Also maxpooling dimension is chosen to improove the span of the second-layer kernel, to be able to determine less localised features.

Poor performance seen, minimum loss of around 0.01...

```ruby
image_size=128 #square images considered (imagesize x imageize)
n_in_channels=2 #Images taken as input onto the CNN


#Below the channels per layer

#Layer 1
n_indiv_channels_c1=8 ; n_indiv_channels_c2=32 
kernel_c1= 8           ; kernel_c2= 8
max_pool_1=4
#Layer2
n_indiv_channels_c3=32 ; n_indiv_channels_c4=32
kernel_c3= 4           ;    kernel_c4= 4 
max_pool_2=4
#Dense layers
p_dropout=0.2
n_neurons=1000

#Dataset generation related

n_batch=400 #Determines minibatch size (number of samples the CNN runs over in one epoch), neeeds to be smaller than the traininig set.
training_set_fraction=0.8
validation_set_fraction=0.2
learning_rate=0.0001
```

## V3 With ReLu, apparently previous CNNs did not haev relu from the iteration 3...

### Version V4

Read a bit more about CNN training
https://github.com/Saibhossain/pytorch-foundations-and-projects/blob/main/Visualizing_and_interpreting.ipynb
https://arxiv.org/abs/1512.03385 -- Resnet achitecture nice to know bout for similarlty sized samples



