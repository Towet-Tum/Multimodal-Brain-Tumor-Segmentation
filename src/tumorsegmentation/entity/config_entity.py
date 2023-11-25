from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    train_URL: str
    val_URL: str
    unzip_dir: Path


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    img_dir : Path 
    mask_dir : Path 
    dataset : Path 
    splited_dataset : Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    train_img_dir : Path
    train_mask_dir : Path
    val_img_dir : Path
    val_mask_dir : Path 
    IMG_HEIGHT : int
    IMG_WIDTH : int
    IMG_DEPTH : int
    IMG_CHANNELS : int
    num_classes : int
    epochs : int
    wt : float
    LR : float
    optim : str 
    batch_size : int
    