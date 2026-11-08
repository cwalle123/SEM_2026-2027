# Convolutional neural network iteration I

> [!NOTE]
> A new version is being worked on which accounts for a focus series of PSFs, improving the performance significantly.

This initial iteration considers a single channel input (i.e. a single PSF) and is trained over a specified of epochs `n_epochs`. The current structure is the one as follows
```ruby
class CNN(torch.nn.Module):
    def __init__(self):

        super(CNN,self).__init__()
        self.conv_pass_1=torch.nn.Sequential(torch.nn.Conv2d(1,16,(9,9),stride=1,padding="same"),torch.nn.ReLU(),torch.nn.Conv2d(16,32,(5,5),padding="same"),torch.nn.ReLU(),torch.nn.MaxPool2d(kernel_size=2,stride=2)) #Yields 64x64 @ 32
        self.conv_pass_2=torch.nn.Sequential(torch.nn.Conv2d(32,32,(3,3),padding="same"),torch.nn.ReLU(),torch.nn.Conv2d(32,16,(3,3),padding="same"),torch.nn.ReLU(),torch.nn.MaxPool2d(kernel_size=2,stride=2)) #Yields  32 @ 32x32 
        #self.conv_pass_2=torch.nn.Sequential(torch.nn.Conv2d(64,32,(3,3)stride=1,padding="same"),)
        self.dense=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Dropout(p=0.1),torch.nn.Linear(16384,1000),torch.nn.ReLU(),torch.nn.Linear(1000,10)) #All follows from max pooling reaulting size and number of channels
    
    def forward(self,x):
        #For forward pass
        x=self.conv_pass_1(x)
        x=self.conv_pass_2(x)
        x=self.dense(x)
        return x
```    

The script allows for analysis of the model on a certain specified dataset, that for now is the default one used for partially training (refer to the OneDrive Datasets for iteration 1)
