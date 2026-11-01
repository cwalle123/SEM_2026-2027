import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path
import os
###################################
#Function and classes definitions
###################################

#CNN class defin
class CNN(torch.nn.Module):
    def __init__(self):

        super(CNN,self).__init__()
        self.conv_pass_1=torch.nn.Sequential(torch.nn.Conv2d(1,16,(3,3),stride=1,padding="same"),torch.nn.ReLU(),torch.nn.Conv2d(16,32,(3,3),padding="same"),torch.nn.ReLU(),torch.nn.MaxPool2d(kernel_size=4,stride=4)) #Yields 32x32 @ 32
        self.conv_pass_2=torch.nn.Sequential(torch.nn.Conv2d(32,32,(3,3),padding="same"),torch.nn.ReLU(),torch.nn.Conv2d(32,32,(3,3),padding="same"),torch.nn.ReLU(),torch.nn.MaxPool2d(kernel_size=2,stride=2)) #Yields  32 @ 16x16 
        #self.conv_pass_2=torch.nn.Sequential(torch.nn.Conv2d(64,32,(3,3)stride=1,padding="same"),)
        self.dense=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Dropout(p=0.1),torch.nn.Linear(8192,1000),torch.nn.ReLU(),torch.nn.Linear(1000,10)) #All follows from max pooling reaulting size and number of channels
    
    def forward(self,x):
        #For forward pass
        x=self.conv_pass_1(x)
        x=self.conv_pass_2(x)
        x=self.dense(x)
        return x 
    

####################################
#Hyperparameters and setup
####################################

#Functionalities
model_name="CNN_iteration_1_v1_normalised"
train=True #To determine weights and saves the parameters
evaluation_mode=False #Loads parameters and used for model inference

n_display=10#epochs between prints of losses display
#Dataset-format related
image_size=128
file_name="psf_dataset.npz" #name of the dataset you want to load

#Learning hyperparameters
n_epochs=100
training_set_fraction=0.8
validation_set_fraction=0.2
learning_rate=0.01

model=CNN()
lossfn=torch.nn.MSELoss(reduction="sum") #Loss function class employed
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate) #Optimiser class used for weight updates


#####################
#Dataset generation
#####################

if train:

    #Acces the data and get what is relevant for training. It becomes easier to do the split here, so the indexes geerated from the file will for now be ingored

    Base_Directory=Path(__file__).resolve().parent.parent #gets the git root to access other folders
    dataset=np.load(f"{Base_Directory}/PSF_from_image/Data/{file_name}")

    images=dataset["images"]
    
    vectors=dataset["labels"]
    ranges=dataset["ranges"]

    if np.shape(images)[0]!=np.shape(vectors)[0]:
        raise ValueError("Images and vector sizes are uncompatible")

    dataset_size=np.shape(images)[0]
    #print(np.shape(vectors))
    print(np.shape(images))
    images=np.reshape(images,(dataset_size,1,128,128)) #For it to not be confused with channel size (N,C,W,H) is used for pytorch
    print(np.shape(images))
    print(np.shape(vectors))
    #Will now split into validation and train (as all figures are randomly generated, then there is no issue with this splitting being biased)

    x_train=images[:int(dataset_size*training_set_fraction)]
    x_validation=images[int(dataset_size*validation_set_fraction):]
    y_train=vectors[:int(dataset_size*training_set_fraction)]
    y_validation=vectors[int(dataset_size*validation_set_fraction):]

    #And now convert to tensor

    x_train=torch.tensor(x_train).to(torch.float32)
    x_validation=torch.tensor(x_validation).to(torch.float32)
    y_train=torch.tensor(y_train).to(torch.float32)
    y_validation=torch.tensor(y_validation).to(torch.float32)


############
#Training
############

os.makedirs(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained_Models/{model_name}",exist_ok=True)

if train:

    loss_train_arr=np.zeros(n_epochs)
    #loss_validation_arr=np.zeros(n_epochs)
    #Loop n_epoch times over the batch
    
    for n in tqdm.tqdm(range(n_epochs)):

        optimizer.zero_grad() #Makes sure all grads for lerning are set to zero

        y_pred=model(x_train)
        loss=lossfn(y_pred,y_train)

        #y_validation=model(x_validation)
        #loss_validation=0

        if n%n_display==0: #To get an idea on training as it goes
            print(f'--------------\nCurrent epoch: {n}\nTraining_loss={loss.item()}')

        loss.backward()
        optimizer.step() #From the loss, the values of the gradient are obtained and stored in the y_pred, then they are updated with  optimiser.step with the learning rule determined earlier


        #Note down the validation and train losses for further evaluation
        loss_train_arr[n]=loss.item()
        #loss_validation_arr[n]=loss_validation.item()


    #Then let's save the model weights and save some plots for quick evaluation of the loss vs iter.
    torch.save(model.state_dict(),f"{Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained_Models/{model_name}/{model_name}.pth")
    np.savez_compressed(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained_Models/{model_name}/model_data.npz",loss_train_arr=loss_train_arr,ranges=ranges)

    print("end of traning. plotting...")
    #plt.plot(range(n_epochs),loss_validation_arr,label="validation losses")
    plt.plot(range(n_epochs),loss_train_arr,label="traning losses")
    plt.legend()
    plt.show()
    #plt.savefig(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained_Models/{model_name}/plotlosses.png")



######################################
#Plotting and evaluation of the model
#######################################

if evaluation_mode:
    #Evaluate validation vs training and first convolutional layers to check for generalisation perfromance 
    model.load_state_dict(torch.load("{Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained Models/{model_name}/{model_name}.pth",weights_only=True))
    model_data=np.load("{Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained_Models/{model_name}/model_data.npz")
    model.eval() #Evaluation mode, dropout set to zero now
    ranges=model_data["ranges"] #use to scale the outputs as the network as trained on normalised labels for unbiased training
    #Space for analysis of the model















