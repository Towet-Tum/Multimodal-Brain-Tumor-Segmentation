import os
from tumorsegmentation.constants import *
from tumorsegmentation.utils.common import read_yaml, create_directories
from tumorsegmentation.entity.config_entity import (DataIngestionConfig, 
                                                    DataPreprocessingConfig,
                                                    TrainingConfig)



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            train_URL=config.train_URL,
            val_URL=config.val_URL,
            
        
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        create_directories([config.root_dir])
        create_directories([config.img_dir, config.mask_dir, config.splited_dataset])

        dataset = os.path.join("artifacts", "data_ingestion", "raw_dataset", "BraTS20Dataset", "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")
        
        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            img_dir=config.img_dir,
            mask_dir=config.mask_dir,
            dataset=Path(dataset),
            splited_dataset=config.splited_dataset,
            
        )
        return data_preprocessing_config
    

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        processed_data = self.config.data_preprocessing
        params = self.params
        create_directories([
            Path(training.root_dir)
        ])
        train_img_dir = os.path.join("artifacts", "preprocessed_data", "train_val_dataset", "train", "images/")
        train_mask_dir = os.path.join("artifacts", "preprocessed_data", "train_val_dataset", "train", "masks/")
        val_img_dir = os.path.join("artifacts", "preprocessed_data", "train_val_dataset", "val", "images/")
        val_mask_dir = os.path.join("artifacts", "preprocessed_data", "train_val_dataset", "val", "masks/")

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            
            
            train_img_dir = str(train_img_dir),
            train_mask_dir = str(train_mask_dir),
            val_img_dir = str(val_img_dir), 
            val_mask_dir = str(val_mask_dir), 

            epochs=params.epochs,
            num_classes=params.num_classes,
            IMG_CHANNELS=params.IMG_CHANNELS,
            IMG_DEPTH=params.IMG_DEPTH,
            IMG_HEIGHT=params.IMG_HEIGHT,
            IMG_WIDTH=params.IMG_WIDTH,
            
            wt= params.wt,
            LR = params.LR,
            optim = params.optim,
            batch_size = params.batch_size,
            
        )

        return training_config