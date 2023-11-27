from tumorsegmentation import logger 
from tumorsegmentation.config.configuration import ConfigurationManager
from tumorsegmentation.components.data_preprocessing import DataPreprocessing



STAGE_NAME = "Data Preprocessing"
class DataPreprocessingPipeline:
    def __init__(self):
        pass 

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.train_data_preprocessing()
        data_preprocessing.train_val_split()
        data_preprocessing.test_data_preprocessing()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> The stage {STAGE_NAME} has started >>>>>>>>>>")
        data_preprocess = DataPreprocessingPipeline()
        data_preprocess.main()
        logger.info(f"The stage {STAGE_NAME} has completed successfully >>>>>>>>>>")
    except Exception as e:
        logger.exception(e)
        raise e