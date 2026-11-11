import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path
import os


####################################
#Hyperparameters and setup
####################################
np.random.seed(1)

#Functionalities
train=True #To determine weights and saves the parameters
evaluation_mode=False #Loads parameters and used for model inference
dataset_analysis_mode=False #Used for evaluation of the dataset fed into the CNN

#Input-putput_details
n_predicted_coeffs=9 #Dimensionality of the output vector (for now this is either 3,7 or 9)
model_name=f"CNN_iteration_2_v2_{n_predicted_coeffs}"

image_size=128 #square images considered (imagesize x imageize)
n_in_channels=2 #Images taken as input onto the CNN
indices_of_interest=[0,2] #Selects the images of onterest taken of the same PSF for training and evalutaion from that
file_name="psf_9coef_A1-B2-A2-C3-A3_n1000.npz" #name of the dataset you want to load

#Other
n_datapoints_plotting = 100 #Number of datapoints used for plotting in total
n_display=10#epochs between prints of losses display
n_batch=300 #Determines batch size (number of samples the CNN runs over in one epoch), neeeds to be smaller than the traininig set.
n_lowpass=50 #Window used for early validation stop whether the general tendency is to increaso er decrease despite noise
n_save=100 #Epochs per save made
n_epochs=5000

#Learning hyperparameters
max_pool_1=2
max_pool_2=2
training_set_fraction=0.8
validation_set_fraction=0.2
learning_rate=0.001

###################################
#Function and classes definitions
###################################

#CNN class definition
class CNN(torch.nn.Module):
    def __init__(self):

        super(CNN,self).__init__()
        self.conv_pass_1=torch.nn.Sequential(torch.nn.Conv2d(n_in_channels,16,(9,9),stride=1,padding="same"),torch.nn.ReLU(),torch.nn.Conv2d(16,32,(5,5),stride=1,padding="same"),torch.nn.ReLU(),torch.nn.MaxPool2d(max_pool_1)) #Yields 64x64 @ 32
        self.conv_pass_2=torch.nn.Sequential(torch.nn.Conv2d(32,32,(3,3),padding="same"),torch.nn.ReLU(),torch.nn.Conv2d(32,16,(3,3),padding="same"),torch.nn.ReLU(),torch.nn.MaxPool2d(max_pool_2)) #Yields  16 @ 32x32 
        #self.conv_pass_2=torch.nn.Sequential(torch.nn.Conv2d(64,32,(3,3)stride=1,padding="same"),)
        self.dense=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Dropout(p=0.1),torch.nn.Linear(int(16*((image_size*image_size)/(max_pool_2**2*max_pool_1**2))),1000),torch.nn.ReLU(),torch.nn.Linear(1000,n_predicted_coeffs)) #All follows from max pooling reaulting size and number of channels
    
    def forward(self,x):
        #For forward pass
        x=self.conv_pass_1(x)
        x=self.conv_pass_2(x)
        x=self.dense(x)
        return x 
    


###Creation of the CNN object instance


model=CNN()
lossfn=torch.nn.MSELoss(reduction="mean") #Loss function class employed
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate) #Optimiser class used for weight updates



#####################
#File conditioning
#####################

#Reqired initialisation for accessing files

Base_Directory=Path(__file__).resolve().parent.parent #gets the git root to access other folders
exists=os.path.isdir(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.2/Trained_Models/{model_name}")

if exists and train:
    yn=input(f"This training will overwrite all existing data on the folder corresponding to {model_name}, are you sure you yould like to proceed? (y/n)\n")
    if yn=="n":
        print("Aborting training...")
        train=False

elif not exists and train:
    yn=input(f"This training will create a new folder corresponding to {model_name}, would you yould like to proceed? (y/n)\n")
    if yn=="y":
        os.makedirs(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.2/Trained_Models/{model_name}",exist_ok=True) #Creates folder for all data to be stored/or to acces for  model evaluation
        print(f"Created new folder in {Base_Directory}/ConvolutionalNeural_Net_Iter.2/Trained_Models/{model_name}")
    else: 
        train=False


#####################
#Dataset generation
#####################


if train:

    #Acces the data and get what is relevant for training. It becomes easier to do the split here, so the indexes geerated from the file will for now be ingored
    
    dataset=np.load(f"{Base_Directory}/PSF_from_image/Data/{file_name}")

    images=dataset["images"]
    images=images[:,[indices_of_interest[0],indices_of_interest[1]]]
    images_std=np.std(images) #In order to standarise
    images=images/images_std#standarisation of the input

    vectors=dataset["labels"]
    ranges=dataset["bounds"]

    if np.shape(images)[0]!=np.shape(vectors)[0]:
        raise ValueError("Images and vector sizes are uncompatible")

    dataset_size=np.shape(images)[0]
    #print(np.shape(vectors))

    images=np.reshape(images,(dataset_size,n_in_channels,image_size,image_size)) #For it to not be confused with channel size (N,C,W,H) is used for pytorch

    #Will now split into validation and train (as all figures are randomly generated, then there is no issue with this splitting being biased)

    x_train=images[:int(dataset_size*training_set_fraction)]
    x_validation=images[int(dataset_size*training_set_fraction):]

    y_train=vectors[:int(dataset_size*training_set_fraction)]
    y_validation=vectors[int(dataset_size*training_set_fraction):]

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

        if n%n_save==0 and n>1: #TO save progress avoids issues when curing mid training
            print(f"\nSaved model and other relevant data at epoch {n}")
            np.savez_compressed(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.2/Trained_Models/{model_name}/model_data.npz",loss_train_arr=loss_train_arr,loss_validation_arr=loss_validation_arr,ranges=ranges,images_std=images_std)
            torch.save(model.state_dict(),f"{Base_Directory}/ConvolutionalNeural_Net_Iter.2/Trained_Models/{model_name}/{model_name}.pth")


        idx=np.random.randint(0,len(x_train)-1,n_batch) #Generates the random indexes for the microbatch generation, otheriwse memory kills my laptop
        
        x_train_batch=x_train[idx]
        y_train_batch=y_train[idx]



        y_pred=model(x_train_batch)
        loss=lossfn(y_pred,y_train_batch)

        y_pred_validation=model(x_validation) #Simply check the validation losses during training
        loss_validation=lossfn(y_pred_validation,y_validation)

        if n%n_display==0: #To get an idea on training as it goes
            print(f'--------------\nCurrent epoch: {n}\nTraining_loss={loss.item()}\nValidation_loss={loss_validation.item()}')
        
        
        loss.backward()
        optimizer.step() #From the loss, the values of the gradient are obtained and stored in the y_pred, then they are updated with  optimiser.step with the learning rule determined earlier


        #Note down the validation and train losses for further evaluation
        loss_train_arr[n]=loss.item()
        loss_validation_arr[n]=loss_validation.item()

        if n%n_lowpass==0 and n>2.5*n_lowpass: #Check validation error running average whether it goes down

            mean1=torch.mean(y_validation[y_validation!=0][-2*n_display:-n_display])
            mean2=torch.mean(y_validation[y_validation!=0][-n_display:])
        
            if mean2>mean1:
                print("Validation increases again... Finishing training")
                break



    print("end of training epochs... moving to file saving and plotting")

    #Then let's save the model weights and save some plots for quick evaluation of the loss vs iter.

    
    print("Saving files")
    torch.save(model.state_dict(),f"{Base_Directory}/ConvolutionalNeural_Net_Iter.2/Trained_Models/{model_name}/{model_name}.pth")
    np.savez_compressed(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.2/Trained_Models/{model_name}/model_data.npz",loss_train_arr=loss_train_arr,loss_validation_arr=loss_validation_arr,ranges=ranges,images_std=images_std)
    print("Saved files")

    print("Plotting...")
    step_plotting=int(len(loss_train_arr)/n_datapoints_plotting)
    plt.plot(range(n_epochs),loss_validation_arr,label="validation losses")
    plt.plot(range(n_epochs),loss_train_arr,label="traning losses")
    plt.legend()
    plt.savefig(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.2/Trained_Models/{model_name}/plotlosses.png")
    plt.show()
    



######################################
#Plotting and evaluation of the model
#######################################

singleanalysis=True
valanalysis=False

if evaluation_mode:

    #Evaluate validation vs training and first convolutional layers to check for generalisation perfromance 
    model_data=np.load(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.2/Trained_Models/{model_name}/model_data.npz")
    model.load_state_dict(torch.load(rf"{Base_Directory}/ConvolutionalNeural_Net_Iter.2/Trained_Models/{model_name}/{model_name}.pth",weights_only=True))
    model.eval() #Evaluation mode, dropout set to zero now


    ranges=model_data["ranges"] #use to scale the outputs as the network as trained on normalised labels for unbiased training
    sigma=model_data["images_std"] #For scaling the input
    
    #Loading of dataset for evalutaion

    dataset=np.load(f"{Base_Directory}/PSF_from_image/Data/{file_name}")
    images=dataset["images"][:,[0,2]]    
    images=torch.tensor(images/sigma).to(torch.float32) #Range from 0,10, to attain the normalised dataset used for training and also makes is suitable format (pytorch used 32bit by default)
    vectors=torch.tensor(dataset["labels"]).to(torch.float32)
    ranges=torch.tensor(dataset["bounds"]).to(torch.float32)
    dataset_size=np.shape(images)[0]
    
    if valanalysis:

        ###Validation analysis of the model

        imidx=int(training_set_fraction*dataset_size) #Image index aftwe which the validation set starts
    
        input=torch.Tensor.reshape(images[imidx:],(int(validation_set_fraction*dataset_size),n_in_channels,image_size,image_size)) #Proper conditioning for the CNN with the corresponding validation batch
        out=ranges[:,0]+(model(input)+1)/2*(ranges[:,1]-ranges[:,0]) #" to convert predicted labels back to meters: low + (labels + 1) / 2 * (high - low)"
        ref=ranges[:,0]+(vectors[:imidx]+1)/2*(ranges[:,1]-ranges[:,0])

        
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
        print(np.shape(images))
        imidx=810 #index of interest
        
        input=torch.Tensor.reshape(images[imidx],(1,n_in_channels,image_size,image_size)) #Proper conditioning for the CNN with the corresponding validation batch
        out=(model(input))[0]
        
        print(ranges[:,0])
        print(ranges[:,1])

        out=(ranges[:,0]+(out+1)/2*(ranges[:,1]-ranges[:,0])).detach().numpy()
        ref=((ranges[:,0]+(vectors[imidx]+1)/2*(ranges[:,1]-ranges[:,0]))).detach().numpy()
        print(out)
        print(ref)


        #Plotting results nicely
        vector_values_labels=np.array(["A1r", "A1i", "B2r", "B2i", "A2r", "A2i", "C3", "A3r", "A3i"])
        x=np.array(range(len(vector_values_labels)))

        plt.suptitle(f"{model_name} evaluation on index {imidx}")

        plt.subplot(131)
        plt.imshow(images[imidx][0])
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(images[imidx][1])
        plt.colorbar()
        plt.subplot(133)
        plt.xticks(x,vector_values_labels)
        #plt.bar(x-0.1,relative_error,width=0.2)
        
        plt.bar(x,(out-ref)/ref*100,width=0.2)
        #plt.bar(x,ref/ref)
        plt.ylabel("Relative error [%]")
        plt.show()
    



if dataset_analysis_mode:
    #Used for aspects of the dataset imported

    idx=200 #Index of PSF to analyse


    dataset=np.load(f"{Base_Directory}/PSF_from_image/Data/{file_name}")
   
    images=dataset["images"][:,[0,2]]
    images=images/np.std(images)
    vectors=dataset["labels"]
    ranges=dataset["bounds"]

    #print(np.shape(images))
    #print(images[1])
    plt.subplot(121)
    plt.imshow(images[idx][0],cmap="viridis") #Just to visualise the PSFs 
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(images[idx][1],cmap="viridis") #Just to visualise the PSFs 
    plt.colorbar()
    plt.show()




