#%%
import glob
from pyexpat import model
import random
import os
import numpy as np
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from moonshine.preprocessing import get_preprocessing_fn
from Classifier import Classifier
from Trainer import Trainer

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from datasets import load_metric
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import random
from collections import OrderedDict
import os 
import cv2
import numpy as np
import glob as gb
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import torchdata.datapipes as dp
import ColorOfArrays
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from moonshine.preprocessing import get_preprocessing_fn
from CustomDataset import CustomDataset
torch.set_float32_matmul_precision("high")
#%%




scaler = MinMaxScaler()


patch_size = 256
root_directory = 'C:\\Users\\Beytullah\\Desktop\\Semantic segmentation dataset'


def print_files():
    for dirname, _, filenames in os.walk('C:\\Users\\Beytullah\\Desktop\\Semantic segmentation dataset'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

def patchify_train_images(root_directory:str):
    image_dataset=[]
    for path, subdirs, files in os.walk(root_directory):
    #print(path)  
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':   #Find all 'images' directories
            images = os.listdir(path)  #List of all image names in this subdirectory
            for i, image_name in enumerate(images):  
                if image_name.endswith(".jpg"):   #Only read jpg images...
               
                    image = cv2.imread(path+"/"+image_name, 1)  #Read each image as BGR
                    SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                    SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                    image = Image.fromarray(image)
                    image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                    #image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                    image = np.array(image)             
       
                    #Extract patches from each image
                    print("Now patchifying image:", path+"/"+image_name)
                    patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
        
                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                        
                            single_patch_img = patches_img[i,j,:,:]
                        
                            #Use minmaxscaler instead of just dividing by 255. 
                            single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                        
                            #single_patch_img = (single_patch_img.astype('float32')) / 255. 
                            single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                            image_dataset.append(single_patch_img)

    return image_dataset

def patchify_mask_images(root_directory:str):
    mask_dataset=[]
    for path, subdirs, files in os.walk(root_directory):
        #print(path)  
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'masks':   #Find all 'images' directories
            masks = os.listdir(path)  #List of all image names in this subdirectory
            for i, mask_name in enumerate(masks):  
                if mask_name.endswith(".png"):   #Only read png images... (masks in this dataset)
               
                    mask = cv2.imread(path+"/"+mask_name, 1)  #Read each image as Grey (or color but remember to map each color to an integer)
                    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                    SIZE_X = (mask.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                    SIZE_Y = (mask.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                    mask = Image.fromarray(mask)
                    mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                    #mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                    mask = np.array(mask)             
       
                    #Extract patches from each image
                    print("Now patchifying mask:", path+"/"+mask_name)
                    patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
        
                    for i in range(patches_mask.shape[0]):
                        for j in range(patches_mask.shape[1]):
                        
                            single_patch_mask = patches_mask[i,j,:,:]
                            #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                            single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.                               
                            mask_dataset.append(single_patch_mask)

    return mask_dataset

def visualize_example_pair(image_dataset,mask_dataset):
    rand_image_number=random.randint(0,len(dataset) )
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(np.reshape(image_dataset[rand_image_number], (patch_size, patch_size, 3)))
    plt.subplot(122)
    plt.imshow(np.reshape(mask_dataset[rand_image_number], (patch_size, patch_size, 3)))
    plt.show()
    



def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label == ColorOfArrays.Building,axis=-1)] = 0
    label_seg [np.all(label==ColorOfArrays.Land,axis=-1)] = 1
    label_seg [np.all(label==ColorOfArrays.Road,axis=-1)] = 2
    label_seg [np.all(label==ColorOfArrays.Vegetation,axis=-1)] = 3
    label_seg [np.all(label==ColorOfArrays.Water,axis=-1)] = 4
    label_seg [np.all(label==ColorOfArrays.Unlabeled,axis=-1)] = 5
    
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg

    

def run_experiment(pretrain=False):

    pretrain_key="pretrained" if pretrain else "scratch"
    exp_name = f"building_model_both_{pretrain_key}"

    #tensorboard to log our experiments
    logger = pl.loggers.TensorBoardLogger("tb_logs", name=exp_name)
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=50,
        enable_progress_bar=True,
        logger=logger,
    )

    model=Classifier(pretrained=pretrain)
    pytrain=Trainer(model)

    #trainer.fit(
    #    model=pytrain,
    #    train_dataloaders=,
    #    val_dataloaders=
    #    )
    
        
def get_dataset(dataset,split):
    tfx=[
        A.RandomCrop(96, 96)
        ]
    train_tfx=[
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
        ]
    
    if split == "train":
        tfx.extend(train_tfx)
        
    tfx.append(ToTensorV2(transpose_mask=True))
    
    A.Compose(tfx)
    
    
    return DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=(split == "train"),
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    
#%%

print_files()
image_dataset=patchify_train_images(root_directory)
mask_dataset=patchify_mask_images(root_directory)


print("Shape of the mask_dataset:{}".format(len(mask_dataset)))
print("Shape of the image_dataset:{}".format(len(image_dataset)))
#%%
image_dataset = np.array(image_dataset)
print(image_dataset.shape)
mask_dataset =  np.array(mask_dataset)
print(mask_dataset.shape)
labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)    

labels = np.array(labels)  

labels = np.expand_dims(labels, axis=3)

print("Unique labels in label dataset are: ", np.unique(labels))

NUM_OF_CLASSES=len(np.unique(labels))

id2label={
     0:"Building",
     1:"Land",
     2:"Road",
     3:"Vegatiation",
     4:"Water",
     5:"Unlabeled"
}

#%%preprocess data

fn = get_preprocessing_fn(model="unet", dataset="fmow_rgb")
image_data_preprocessed=fn(image_dataset)
image_data_preprocessed=image_data_preprocessed.astype(np.float32)

print(mask_dataset)
print(image_data_preprocessed)
print(labels)

print(f"Shape of the preprocessed image dataset {image_data_preprocessed.shape}")


#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_data_preprocessed, labels, test_size = 0.20, random_state = 42)

train_set=CustomDataset(X_train, y_train, True)
test_set=CustomDataset(X_test, y_test,False)


train_loader=DataLoader(train_set,batch_size=16,shuffle=True,drop_last=True,num_workers=4,pin_memory=True)
valid_loader=DataLoader(test_set,batch_size=16,drop_last=True,num_workers=4,pin_memory=True)

#%%
def run_experiment(pretrain=False):
    """Run an experiment with or without pretraining"""
    # Download the files into a folder with 8band and geojson subfolders.
    #files = glob.glob(os.path.join(root_directory))

    # Create a name for Tensorboard
    pretrain_key = "pretrained" if pretrain else "scratch"
    exp_name = f"building_model_both_{pretrain_key}"

    # Create our datasets.
    
    train_dataset = train_loader
    val_dataset = valid_loader

    # We'll use Tensorboard to log our experiments, but this is optional.
    logger = pl.loggers.TensorBoardLogger("tb_logs", name=exp_name)
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=50,
        enable_progress_bar=True,
        logger=logger,
    )

    # We'll use the custom lightning trainer, confusingly called a model by the lightning API.
    model = Classifier(pretrained=pretrain)
    pytrain = Trainer(model)

    # Train!
    trainer.fit(
        model=pytrain,
        train_dataloaders=train_dataset,
        val_dataloaders=val_dataset,
    )

print("Run experiment with pretraining")
run_experiment(pretrain=True)
