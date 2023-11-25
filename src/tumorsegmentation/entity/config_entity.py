from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    train_URL: str
    val_URL: str
    unzip_dir: Path