# Convolutional Neural Network V2

This CNN code allows for easier tuning of hyperparameters as well as GPU acceleration with Nvidia's CUDA platform.
In order to run the code with CUDA follow the tutorial [here](https://www.geeksforgeeks.org/deep-learning/how-to-use-gpu-acceleration-in-pytorch/)


## Versions 1-3 have all ben using the structure below:

for V1-2 p_drpopout=0.2
for V3 p_dropout =0

- All yielded vaidation losses around 0.01 for minimum validation losses mainly independent of p_droppout and other opimisation parameters. So similar optimal performance after training
- Only difference betweek v1 and v2 is the number of epochs used for training

```ruby
#CNN class definition
class CNN(torch.nn.Module):
    def __init__(self):

        super(CNN,self).__init__()
        self.conv_pass_1=torch.nn.Sequential(torch.nn.Conv2d(n_in_channels,16,(9,9),stride=1,padding="same"),torch.nn.ReLU(),torch.nn.Conv2d(16,32,(5,5),stride=1,padding="same"),torch.nn.ReLU(),torch.nn.MaxPool2d(max_pool_1)) #Yields 64x64 @ 32
        self.conv_pass_2=torch.nn.Sequential(torch.nn.Conv2d(32,32,(3,3),padding="same"),torch.nn.ReLU(),torch.nn.Conv2d(32,16,(3,3),padding="same"),torch.nn.ReLU(),torch.nn.MaxPool2d(max_pool_2)) #Yields  16 @ 32x32 
        #self.conv_pass_2=torch.nn.Sequential(torch.nn.Conv2d(64,32,(3,3)stride=1,padding="same"),)
        self.dense=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Dropout(p=0),torch.nn.Linear(int(16*((image_size*image_size)/(max_pool_2**2*max_pool_1**2))),1000),torch.nn.ReLU(),torch.nn.Linear(1000,n_predicted_coeffs)) #All follows from max pooling reaulting size and number of channels
    
    def forward(self,x):
        #For forward pass
        x=self.conv_pass_1(x)
        x=self.conv_pass_2(x)
        x=self.dense(x)
        return x 
    
```

other parameters used: 

```ruby
#Other
n_datapoints_plotting = 100 #Number of datapoints used for plotting in total
n_display=10#epochs between prints of losses display
n_batch=400 #Determines batch size (number of samples the CNN runs over in one epoch), neeeds to be smaller than the traininig set.
n_lowpass=50 #Window used for early validation stop whether the general tendency is to increaso er decrease despite noise
n_save=100 #Epochs per save made
n_epochs=5000

#Learning hyperparameters
max_pool_1=2
max_pool_2=2
training_set_fraction=0.8
validation_set_fraction=0.2
learning_rate=0.001
```

# Examples of V3 version results on _validaton_ set
<img width="1600" height="759" alt="image" src="https://github.com/user-attachments/assets/6dc09c25-4b43-4a17-841d-2b10c3dd641c" />
<img width="1600" height="739" alt="image" src="https://github.com/user-attachments/assets/c38b4b1c-6089-401e-8cee-770229082e18" />

