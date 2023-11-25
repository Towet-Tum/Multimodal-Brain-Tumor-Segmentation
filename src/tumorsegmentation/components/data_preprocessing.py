import os
import glob
import numpy as np
import splitfolders
import nibabel as nib 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tumorsegmentation.entity.config_entity import DataPreprocessingConfig



class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def train_data_preprocessing(self):
        self.t2_list = sorted(glob.glob(f"{self.config.dataset}/*/*t2.nii"))
        self.t1ce_list = sorted(glob.glob(f"{self.config.dataset}/*/*t1ce.nii"))
        self.flair_list = sorted(glob.glob(f"{self.config.dataset}/*/*flair.nii"))
        self.mask_list = sorted(glob.glob(f"{self.config.dataset}/*/*seg.nii"))
        scaler = MinMaxScaler()

  
        for img in range(len(self.t2_list)):   #Using t1_list as all lists are of same size
            print("Now preparing image and masks number: ", img)

            temp_image_t2=nib.load(self.t2_list[img]).get_fdata()
            temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

            temp_image_t1ce=nib.load(self.t1ce_list[img]).get_fdata()
            temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)

            temp_image_flair=nib.load(self.flair_list[img]).get_fdata()
            temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

            temp_mask=nib.load(self.mask_list[img]).get_fdata()
            temp_mask=temp_mask.astype(np.uint8)
            temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
            #print(np.unique(temp_mask))


            temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)

            #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
            #cropping x, y, and z
            temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
            temp_mask = temp_mask[56:184, 56:184, 13:141]

            val, counts = np.unique(temp_mask, return_counts=True)

            if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
                print("Save Me")
                temp_mask= to_categorical(temp_mask, num_classes=4)
                np.save(f"{self.config.img_dir}/image_"+str(img)+'.npy', temp_combined_images)
                np.save(f"{self.config.mask_dir}/mask_"+str(img)+'.npy', temp_mask)

            else:
                print("I am useless")
        

    def train_val_split(self):
        input_folder = "/home/towet/Desktop/Visions/tumors/Multimodal-Brain-Tumor-Segmentation/artifacts/preprocessed_data/dataset"
        output_folder = self.config.splited_dataset
        splitfolders.ratio(input_folder, output_folder, seed=42, ratio=(.8, 0.2), group_prefix=None)
