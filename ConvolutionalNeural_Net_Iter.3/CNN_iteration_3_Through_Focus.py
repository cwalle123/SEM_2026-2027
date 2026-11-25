
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path
import os

#In order t consider the PSF function packages:
Base_Directory=Path(__file__).resolve().parent.parent #gets the git root to access other folders
sys.path.insert(1, f'{Base_Directory}/PSF_from_Image')
import generate_psf_dataset_NEW as psf_gen #Same generator that made the dataset: channels C1 = +5 um and -5 um, labels = c / (high - low)

####################################
#Hyperparameters and setup
####################################


#####Main parameters###########################################################################################

n_predicted_coeffs=9 #Dimensionality of the output vector (for now this is either 3,7 or 9)
model_name=f"CNN_iteration_3_v1_{n_predicted_coeffs}"
file_name="psf_9coef_A1-B2-A2-C3-A3_n3000.npz" #name of the dataset you want to load
data_folder= f'{Base_Directory}/PSF_from_Image/Data (V2 format)'
indices_of_interest=[0,1] #Selects the images of onterest taken of the same PSF for training and evalutaion from that

###Dataset generation related

training_set_fraction=0.8
validation_set_fraction=0.2
learning_rate=0.0001


###Functionalities

train=False #To determine weights and saves the parameters
evaluation_mode=True #Loads parameters and used for model inference
dataset_analysis_mode=False #Used for evaluation of the dataset fed into the CNN
valanalysis=True

###Validation analyis quick inputs

autosave=True #keeps images
singleanalysis=True #For a single aberration vector to check

a1r = 0;      a1i = 0;            # astigmatism            (e-8)
b2r = 0;      b2i = 0;            # coma                   (e-6)
a2r = 1e-6;  a2i = 1e-6;          # 3-fold astigmatism     (e-6)
c3  = 0e-3                        # spherical aberration   (e-3)
a3r = 0e-6;      a3i = 0e-6;      # 4-fold astigmatism     (e-4)

validation_analysis_vector=np.array([a1r, a1i, b2r, b2i, a2r, a2i, c3, a3r, a3i]) #Same structure as the PSF analysis


#####Structure of the CNN###########################################################################################

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

#####Other###########################################################################################

n_datapoints_plotting = 100 #Number of datapoints used for plotting in total
n_display=10#epochs between prints of losses display
n_batch=400 #Determines minibatch size (number of samples the CNN runs over in one epoch), neeeds to be smaller than the traininig set.
n_lowpass=15000000 #Window used for early validation stop whether the general tendency is to increase or decrease despite noise
n_save=100 #Epochs per save made
n_epochs=5000



###################################
#Function and classes definitions
###################################

#CNN class definition, for this case grouped convolution is used
class CNN(torch.nn.Module):
    def __init__(self):

        super(CNN,self).__init__()
        self.conv_pass_1=torch.nn.Sequential(torch.nn.Conv2d(n_in_channels,n_indiv_channels_c1,kernel_c1,padding="same"),torch.nn.Conv2d(n_indiv_channels_c1,n_indiv_channels_c2,kernel_c2,padding="same"),torch.nn.MaxPool2d(max_pool_1))
        self.conv_pass_2=torch.nn.Sequential(torch.nn.Conv2d(n_indiv_channels_c2,n_indiv_channels_c3,kernel_c3,padding="same"),torch.nn.Conv2d(n_indiv_channels_c3,n_indiv_channels_c4,kernel_c4,padding="same"),torch.nn.MaxPool2d(max_pool_2))
        self.dense=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Dropout(p=p_dropout),torch.nn.Linear(int(n_indiv_channels_c4*((image_size*image_size)/(max_pool_2**2*max_pool_1**2))),n_neurons),torch.nn.ReLU(),torch.nn.Linear(n_neurons,n_predicted_coeffs)) #All follows from max pooling reaulting size and number of channels
    
    def forward(self,x):

        #For forward pass
        x=self.conv_pass_1(x)
        x=self.conv_pass_2(x)
        x=self.dense(x)
        return x 
    
    
#Misc definitions for generating PSF

def load_dataset(file_name):
    """Loads a dataset made by generate_psf_dataset_NEW.py and checks it has the 2 channels and the c / (high - low) labels this CNN expects."""
    dataset=np.load(f'{data_folder}/{file_name}')
    if dataset["images"].shape[1]!=n_in_channels or not np.allclose(dataset["defocus_values"],psf_gen.defocus_values):
        raise ValueError(f"{file_name} has defocus channels {dataset['defocus_values']}, expected {psf_gen.defocus_values}: "
                         "it was made with the old generator, regenerate it with generate_psf_dataset_NEW.py")
    return dataset

def make_cnn_input(aber):
    """Aberration coefficients [A1r, ..., A3i] [m] -> unstandardised CNN input (1, n_in_channels, image_size, image_size) float32 tensor,
    made exactly like the dataset images (same generator, same defocus channels, same crop and binning)."""
    aber=np.asarray(aber,dtype=float)
    outside=(aber<psf_gen.aberration_bounds[:,0])|(aber>psf_gen.aberration_bounds[:,1])
    if outside.any():
        print(f"Warning: {[l for l,o in zip(psf_gen.aberration_labels,outside) if o]} outside the training bounds, the CNN never saw such PSFs")
    vectors=psf_gen.sets_from_shared(aber[None,:])[0] #(n_in_channels, 10): one full vector per defocus channel
    generator=psf_gen.PSFGenerator(psf_gen.V,20e-3,150e-12,3072,1152)
    images=np.stack([psf_gen.bin_image(generator(v),image_size) for v in vectors]).astype(np.float32)
    return torch.from_numpy(images[None])

def labels_to_meters(labels,ranges):
    """Network output / labels -> coefficients [m]: labels * (high - low), the inverse of the generator's labels = c / (high - low)."""
    return labels*(ranges[:,1]-ranges[:,0])


###Creation of the CNN object instance


model=CNN()


lossfn=torch.nn.MSELoss(reduction="mean") #Loss function class employed
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate) #Optimiser class used for weight updates



#####################
#File conditioning
#####################

#Reqired initialisation for accessing files

exists=os.path.isdir(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.3/Trained_Models/{model_name}")

if exists and train:
    yn=input(f"This training will overwrite all existing data on the folder corresponding to {model_name}, are you sure you yould like to proceed? (y/n)\n")
    if yn=="n":
        print("Aborting training...")
        train=False

elif not exists and train:
    yn=input(f"This training will create a new folder corresponding to {model_name}, would you yould like to proceed? (y/n)\n")
    if yn=="y":
        os.makedirs(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.3/Trained_Models/{model_name}",exist_ok=True) #Creates folder for all data to be stored/or to acces for  model evaluation
        print(f"Created new folder in {Base_Directory}/ConvolutionalNeural_Net_Iter.3/Trained_Models/{model_name}")
    else: 
        train=False


#####################
#Dataset generation
#####################


if train:

    #Acces the data and get what is relevant for training. It becomes easier to do the split here, so the indexes geerated from the file will for now be ingored
    
    dataset=load_dataset(file_name)

    images=dataset["images"] #Already only the 2 defocus channels (+5 um, -5 um), no selection needed
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


    #Passes everything to GPU so it is on the same environment as the model, which avoida having to back and froth from CPU to GPU

    x_train=x_train.to('cuda')
    x_validation=x_validation.to('cuda')
    y_train=y_train.to('cuda')
    y_validation=y_validation.to('cuda')

############
#Training
############


if train:
    model=model.to('cuda') #Uses GPU acceleration moves model to GPU :D
    loss_train_arr=np.zeros(n_epochs)
    loss_validation_arr=np.zeros(n_epochs)
    #Loop n_epoch times over the batch
    
    for n in tqdm.tqdm(range(n_epochs)):

        optimizer.zero_grad() #Makes sure all grads for lerning are set to zero

        if n%n_save==0 and n>1: #TO save progress avoids issues when curing mid training
            print(f"\nSaved model and other relevant data at epoch {n}")
            np.savez_compressed(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.3/Trained_Models/{model_name}/model_data.npz",loss_train_arr=loss_train_arr,loss_validation_arr=loss_validation_arr,ranges=ranges,images_std=images_std)
            torch.save(model.state_dict(),f"{Base_Directory}/ConvolutionalNeural_Net_Iter.3/Trained_Models/{model_name}/{model_name}.pth")


        idx=np.random.randint(0,len(x_train)-1,n_batch) #Generates the random indexes for the microbatch generation, otheriwse memory kills my laptop
        
        x_train_batch=x_train[idx]
        y_train_batch=y_train[idx]



        y_pred=model(x_train_batch)
        loss=lossfn(y_pred,y_train_batch)

        with torch.no_grad(): #Validation is only measured, never trained on, so no autograd graph is kept (otherwise it doesn't fit in 6 GB of GPU memory)
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
            
            mean1=np.mean(loss_validation_arr[loss_validation_arr!=0][int(-2*n_display):-n_display])
            mean2=np.mean(loss_validation_arr[loss_validation_arr!=0][-n_display:])
          
            if mean2>mean1:
                print("Validation increases again... Finishing training")
                break



    print("end of training epochs... moving to file saving and plotting")

    #Then let's save the model weights and save some plots for quick evaluation of the loss vs iter.

    
    print("Saving files")
    torch.save(model.state_dict(),f"{Base_Directory}/ConvolutionalNeural_Net_Iter.3/Trained_Models/{model_name}/{model_name}.pth")
    np.savez_compressed(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.3/Trained_Models/{model_name}/model_data.npz",loss_train_arr=loss_train_arr,loss_validation_arr=loss_validation_arr,ranges=ranges,images_std=images_std)
    print("Saved files")

    print("Plotting...")
    step_plotting=int(len(loss_train_arr)/n_datapoints_plotting)
    plt.plot(range(n_epochs),loss_validation_arr,label="validation losses")
    plt.plot(range(n_epochs),loss_train_arr,label="traning losses")
    plt.legend()
    plt.savefig(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.3/Trained_Models/{model_name}/plotlosses.png")
    plt.show()
    



######################################
#Plotting and evaluation of the model
#######################################



if evaluation_mode:

    #Evaluate validation vs training and first convolutional layers to check for generalisation perfromance 
    model_data=np.load(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.3/Trained_Models/{model_name}/model_data.npz")
    model.load_state_dict(torch.load(rf"{Base_Directory}/ConvolutionalNeural_Net_Iter.3/Trained_Models/{model_name}/{model_name}.pth",weights_only=True))
    model.eval() #Evaluation mode, dropout set to zero now


    ranges=model_data["ranges"] #use to scale the outputs as the network as trained on normalised labels for unbiased training
    sigma=model_data["images_std"] #For scaling the input
    
    #Loading of dataset for evalutaion

    dataset=load_dataset(file_name)
    images=dataset["images"] #Only the 2 defocus channels
    images=torch.tensor(images/sigma).to(torch.float32) #Range from 0,10, to attain the normalised dataset used for training and also makes is suitable format (pytorch used 32bit by default)
    vectors=torch.tensor(dataset["labels"]).to(torch.float32)
    ranges=torch.tensor(dataset["bounds"]).to(torch.float32)
    dataset_size=np.shape(images)[0]
    
    if valanalysis:

        ###Validation analysis of the model

        imidx=int(training_set_fraction*dataset_size) #Image index aftwe which the validation set starts
    
        input=torch.Tensor.reshape(images[imidx:],(int(validation_set_fraction*dataset_size),n_in_channels,image_size,image_size)) #Proper conditioning for the CNN with the corresponding validation batch
        out=labels_to_meters(model(input),ranges) #Predicted labels back to meters: labels * (high - low)
        ref=labels_to_meters(vectors[imidx:],ranges) #The validation labels (same samples as the input above)

        
        relative_error=(out-ref)/ref #% Note the size of this, it will be (datasetsize-imidx,Ncoeffs)
        relative_error=relative_error.detach().numpy()
        mean_relative_error=np.mean(relative_error,axis=0)

        plt.suptitle(f"{model_name}'s evaluation on validation set")
        plt.subplot(121)
        plt.bar(dataset["label_names"],mean_relative_error) #The predicted coefficients (no C1, it is set by the defocus channels)
        plt.ylabel("Mean relative eror")
        plt.subplot(122)

        plt.plot(range(np.shape(relative_error)[0]),np.mean(relative_error,axis=1))
        plt.xlabel("Sample index")
        plt.ylabel("Mean of relative error vector")
        
        plt.show()


    if singleanalysis:


    
        ###Analysis of single PSF's performance by the CNN
        os.makedirs(f"{Base_Directory}/ConvolutionalNeural_Net_Iter.3/Analysis",exist_ok=True) 


        input=make_cnn_input(validation_analysis_vector) #Size should be correct, (1,2,128,128) for the resolutions studied
        input=input/sigma #Normalise
        out=(model(input))[0]

        out=labels_to_meters(out,ranges).detach().numpy() # To bring it back to the correct units
        ref=validation_analysis_vector

        reconstruction=make_cnn_input(out)/sigma #Recreated PSF w predicted coeffs for the same through focus
        err_image=np.abs(input-reconstruction)
    
        

        #Plotting results nicely
        np.set_printoptions(linewidth=100,formatter={"float": lambda x: f"{x:.2e}"})
        vector_values_labels=np.array(["A1r", "A1i", "B2r", "B2i", "A2r", "A2i", "C3", "A3r", "A3i"])
        x=np.array(range(len(vector_values_labels)))

        fig = plt.figure(figsize=(12, 6), layout="constrained")


        print(f'Real. vector: {ref}')
        print(f'Pred. vector {out}')
        
        plt.suptitle(f"{model_name} evaluation on predefined vector")

        plt.subplot(131)
        plt.imshow(input[0,0])
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(input[0,1])
        plt.colorbar()
        plt.subplot(133)
        plt.xticks(x,vector_values_labels)
        #plt.bar(x-0.1,relative_error,width=0.2)
        
        plt.bar(x,(out-ref)/ref*100,width=0.2)
        #plt.bar(x,ref/ref)
        plt.ylabel("Relative error [%]")
    
        plt.savefig(f'{Base_Directory}/ConvolutionalNeural_Net_Iter.3/Analysis/{model_name}_{validation_analysis_vector}_1.png')
        plt.show()
        

        
        fig = plt.figure(figsize=(12, 8), layout="constrained") #Plot PSf as built VS real one and check out the redonsrtruction errors visually, both standarised!!
        fig.get_layout_engine().set(rect=(0, 0.12, 1, 0.88))
        

        plt.subplot(231) # The recreated PSF
        plt.title("Recreated overfocus")
        plt.imshow(reconstruction[0,0])      
        plt.colorbar()

        plt.subplot(234) # The recreated PSF
        plt.title("Recreated underfocus")
        plt.imshow(reconstruction[0,1])      
        plt.colorbar()

        plt.subplot(232) # The original PSF series
        plt.title("Original overfocus")
        plt.imshow(input[0,0]) #Overfocused C1=5mum
        plt.colorbar()

        plt.subplot(235) # The original PSF series
        plt.title("Original underfocus")
        plt.imshow(input[0,1]) #Underfocused C1=-5mum
        plt.colorbar()

        plt.subplot(233)
        plt.title("Reconstruction errors overfocused")
        plt.imshow(err_image[0,0],cmap="hot")
        plt.colorbar()

        plt.subplot(236)
        plt.title("Reconstruction errors underfocused")
        plt.imshow(err_image[0,1],cmap="hot")
        plt.colorbar()
        
        #Text alinment and formatting for easier to read in second plot
        def fmt(a, width=10, prec=2):
            return " ".join(f"{x:>{width}.{prec}e}" for x in (a))

        def fmt_labels(labels, width=10):
            return " ".join(f"{l:>{width}}" for l in labels)

        width = 10
        fig.text(0.5, 0.04,
                f"{'':6}{fmt_labels(vector_values_labels, width)}\n" f"Real. {fmt(ref, width)}\n" f"Pred. {fmt(out, width)}", ha="center", family="monospace", multialignment="left")
        
        plt.savefig(f'{Base_Directory}/ConvolutionalNeural_Net_Iter.3/Analysis/{model_name}_{validation_analysis_vector}_2.png')
        plt.show()




if dataset_analysis_mode:
    #Used for aspects of the dataset imported

    idx=200 #Index of PSF to analyse


    dataset=load_dataset(file_name)

    images=dataset["images"] #Only the 2 defocus channels
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




