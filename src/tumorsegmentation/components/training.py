import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tumorsegmentation.utils.common import imageLoader
import segmentation_models_3D as sm 
import numpy as np
from tumorsegmentation.models.unet_3D import unet_3D_model
from tumorsegmentation.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

   
    def train_valid_generator(self):
        train_img_dir = self.config.train_img_dir
        train_mask_dir = self.config.train_mask_dir
        val_img_dir = self.config.val_img_dir
        val_mask_dir = self.config.val_mask_dir
        train_img_list = os.listdir(train_img_dir)
        train_mask_list = os.listdir(train_mask_dir)
        val_img_list = os.listdir(val_img_dir)
        val_mask_list = os.listdir(val_mask_dir)
        batch_size=2
        train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
        val_img_datagen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)
        return train_img_datagen, val_img_datagen, train_img_list, val_img_list
    
    def train(self):
        self.train_img_datagen, self.val_img_datagen, self.train_img_list, self.val_img_list = self.train_valid_generator()
        self.steps_per_epoch = len(self.train_img_list) // self.config.batch_size
        self.val_steps_per_epoch = len(self.val_img_list) // self.config.batch_size
        wt0, wt1, wt2, wt3 = self.config.wt,self.config.wt,self.config.wt,self.config.wt
        dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
        focal_loss = sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)
        metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]
        self.optim = tf.keras.optimizers.Adam(self.config.LR)
        self.model = unet_3D_model(self.config.IMG_HEIGHT,self.config.IMG_WIDTH,
                                        self.config.IMG_DEPTH,self.config.IMG_CHANNELS,
                                        self.config.num_classes)
        self.model.compile(optimizer=self.optim, loss=total_loss, metrics=metrics)
        self.model.fit(self.train_img_datagen, 
                    steps_per_epoch=self.steps_per_epoch,
                        epochs=1,
                        verbose=1, 
                        validation_data=self.val_img_datagen, 
                    validation_steps = self.val_steps_per_epoch)
        self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )