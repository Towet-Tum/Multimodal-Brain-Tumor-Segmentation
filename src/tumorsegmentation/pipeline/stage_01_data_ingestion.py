from tumorsegmentation import logger 
from tumorsegmentation.config.configuration import ConfigurationManager
from tumorsegmentation.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion"
class DataIngestionPipeline:
    def __init__(self):
        pass 

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.extract_zip_file()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>> The stage {STAGE_NAME} has started >>>>>>>")
        data_ingestion = DataIngestionPipeline()
        data_ingestion.main()
        logger.info(f">>>>>>>> The stage {STAGE_NAME} has comleted succefully >>>>>>>")
    except Exception as e:
        logger.exception(e)
        raise e



