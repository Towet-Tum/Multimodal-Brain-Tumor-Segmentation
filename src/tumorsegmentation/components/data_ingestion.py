import os
import urllib.request as request
import zipfile
from tumorsegmentation import logger
from tumorsegmentation.utils.common import get_size
import requests
from tumorsegmentation.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_train_data(self):
        self.local_filename = self.config.train_URL.split('/')[-1]
        # NOTE the stream=True parameter below
        with requests.get(self.config.train_URL, stream=True) as r:
            r.raise_for_status()
            with open(self.local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return self.local_filename
    
    def download_val_data(self):
        self.local_filename = self.config.val_URL.split('/')[-1]
        # NOTE the stream=True parameter below
        with requests.get(self.config.train_URL, stream=True) as r:
            r.raise_for_status()
            with open(self.local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return self.local_filename
    
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.download_train_data(), 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        with zipfile.ZipFile(self.download_val_data(), 'r') as zip_ref:
            zip_ref.extractall(unzip_path)