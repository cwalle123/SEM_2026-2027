import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path
import os

###################################
#Function and classes definitions
###################################

#CNN class definition
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
    

####################################
#Hyperparameters and setup
####################################

#Functionalities
model_name="CNN_iteration_1_v2_normalised"
train=False #To determine weights and saves the parameters
evaluation_mode=True #Loads parameters and used for model inference
dataset_analysis_mode=False #Used for evaluation of the dataset fed into the CNN

n_display=3#epochs between prints of losses display
#Dataset-format related
image_size=128
file_name="psf_dataset.npz" #name of the dataset you want to load
n_datapoints_plotting = 100 #Number of datapoints used for plotting in total

#Learning hyperparameters
n_epochs=500
training_set_fraction=0.8
validation_set_fraction=0.2
learning_rate=0.001

model=CNN()
lossfn=torch.nn.MSELoss(reduction="mean") #Loss function class employed
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate) #Optimiser class used for weight updates


#####################
#File conditioning
#####################

#Reqired initialisation for accessing files

Base_Directory=Path(__file__).resolve().parent.parent #gets the git root to access other folders
exists=os.path.isdir(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained_Models/{model_name}")

if exists and train:
    yn=input(f"This training will overwrite all existing data on the folder corresponding to {model_name}, are you sure you yould like to proceed? (y/n)\n")
    if yn=="n":
        print("Aborting training...")
        train=False

elif not exists and train:
    yn=input(f"This training will create a new folder corresponding to {model_name}, would you yould like to proceed? (y/n)\n")
    if yn=="y":
        os.makedirs(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained_Models/{model_name}",exist_ok=True) #Creates folder for all data to be stored/or to acces for  model evaluation
        print(f"Created new folder in {Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained_Models/{model_name}")
    else: 
        train=False


#####################
#Dataset generation
#####################


if train:

    #Acces the data and get what is relevant for training. It becomes easier to do the split here, so the indexes geerated from the file will for now be ingored
    
    dataset=np.load(f"{Base_Directory}/PSF_from_image/Data/{file_name}")

    images=dataset["images"]
    images=images/np.max(images)*10 #Normalisation
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


if train:

    loss_train_arr=np.zeros(n_epochs)
    loss_validation_arr=np.zeros(n_epochs)
    #Loop n_epoch times over the batch
    
    for n in tqdm.tqdm(range(n_epochs)):

        optimizer.zero_grad() #Makes sure all grads for lerning are set to zero

        y_pred=model(x_train)
        loss=lossfn(y_pred,y_train)

        y_pred_validation=model(x_validation) #Simply check the validation losses during training
        loss_validation=lossfn(y_pred_validation,y_validation)

        if n%n_display==0: #To get an idea on training as it goes
            print(f'--------------\nCurrent epoch: {n}\nTraining_loss={loss.item()}')

        loss.backward()
        optimizer.step() #From the loss, the values of the gradient are obtained and stored in the y_pred, then they are updated with  optimiser.step with the learning rule determined earlier


        #Note down the validation and train losses for further evaluation
        loss_train_arr[n]=loss.item()
        loss_validation_arr[n]=loss_validation.item()


    #Then let's save the model weights and save some plots for quick evaluation of the loss vs iter.
    torch.save(model.state_dict(),f"{Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained_Models/{model_name}/{model_name}.pth")
    np.savez_compressed(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained_Models/{model_name}/model_data.npz",loss_train_arr=loss_train_arr,loss_validation_arr=loss_validation_arr,ranges=ranges)

    print("end of traning. plotting...")
    step_plotting=int(len(loss_train_arr)/n_datapoints_plotting)
    plt.plot(range(n_epochs),loss_validation_arr,label="validation losses")
    plt.plot(range(n_epochs),loss_train_arr,label="traning losses")
    plt.legend()
    plt.show()
    #plt.savefig(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained_Models/{model_name}/plotlosses.png")



######################################
#Plotting and evaluation of the model
#######################################

singleanalysis=True
valanalysis=False



if evaluation_mode:

    #Evaluate validation vs training and first convolutional layers to check for generalisation perfromance 
    model_data=np.load(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained_Models/{model_name}/model_data.npz")
    model.load_state_dict(torch.load(rf"{Base_Directory}/ConvolutionalNeural_Net_Iter.1/Trained_Models/{model_name}/{model_name}.pth",weights_only=True))

    model.eval() #Evaluation mode, dropout set to zero now
    ranges=model_data["ranges"] #use to scale the outputs as the network as trained on normalised labels for unbiased training

    #Loading of dataset for evalutaion

    dataset=np.load(f"{Base_Directory}/PSF_from_image/Data/{file_name}")
    images=dataset["images"]    
    images=torch.tensor(dataset["images"]/(np.max(images)*0.1)).to(torch.float32) #Range from 0,10, to attain the normalised dataset used for training and also makes is suitable format (pytorch used 32bit by default)
    vectors=torch.tensor(dataset["labels"]).to(torch.float32)
    ranges=torch.tensor(dataset["ranges"]).to(torch.float32)
    dataset_size=np.shape(images)[0]
    
    if valanalysis:

        ###Validation analysis of the model

        imidx=int(training_set_fraction*dataset_size) #Image index aftwe which the validation set starts

        input=torch.Tensor.reshape(images[imidx:],(int(validation_set_fraction*dataset_size),1,128,128)) #Proper conditioning for the CNN with the corresponding validation batch
        out=model(input)*ranges
        ref=vectors[imidx:]*ranges

        
        relative_error=(out-ref)/ref*100 #% Note the size of this, it will be (datasetsize-imidx,Ncoeffs)
        relative_error=relative_error.detach().numpy()
        mean_relative_error=np.mean(relative_error,axis=0)

        plt.suptitle(f"{model_name}'s evaluation on validation set")
        plt.subplot(121)
        plt.bar(np.array(["C1", "A1r", "A1i", "B2r", "B2i", "A2r", "A2i", "C3", "A3r", "A3i"]),mean_relative_error)
        plt.ylabel("Mean relative eror")
        plt.subplot(122)

        plt.plot(range(np.shape(relative_error)[0]),np.mean(relative_error,axis=1))
        plt.xlabel("Sample index")
        plt.ylabel("Mean of error vector")
        plt.show()


    if singleanalysis:

        ###Analysis of single PSF's performance by the CNN

        imidx=932 #index of interest
        
        input=torch.Tensor.reshape(images[imidx],(1,1,128,128)) #Proper conditioning for the CNN with the corresponding validation batch
        out=(model(input)*ranges).detach().numpy()
        out=out[0]
        ref=(vectors[imidx]*ranges).detach().numpy()
        relative_error=(out-ref)/ref
        relative_error=relative_error


        #Plotting results nicely
        vector_values_labels=np.array(["C1", "A1r", "A1i", "B2r", "B2i", "A2r", "A2i", "C3", "A3r", "A3i"])
        x=np.array(range(10))

        plt.suptitle(f"{model_name} evaluation on index {imidx}")
        plt.subplot(121)
        plt.imshow(images[imidx])
        plt.subplot(122)
        plt.xticks(x,vector_values_labels)
        #plt.bar(x-0.1,relative_error,width=0.2)
        plt.bar(x,out/ref,width=0.2)
        #plt.bar(x,ref/ref)
        plt.ylabel("Relative outputs [-]")
        plt.show()
    



if dataset_analysis_mode:
    #Used for aspects of the dataset imported

    dataset=np.load(f"{Base_Directory}/PSF_from_image/Data/{file_name}")
   
    images=dataset["images"]    
    images=dataset["images"]/(np.max(images)*0.1) #Range from 0,10
    vectors=dataset["labels"]
    ranges=dataset["ranges"]

    #print(np.shape(images))
    #print(images[1])
    
    plt.imshow(images[100],cmap="viridis") #Just to visualise the PSFs 

    plt.colorbar()
    plt.show()












