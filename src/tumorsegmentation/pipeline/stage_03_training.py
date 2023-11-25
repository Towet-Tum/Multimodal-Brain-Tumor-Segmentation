from tumorsegmentation import logger
from tumorsegmentation.config.configuration import ConfigurationManager
from tumorsegmentation.components.training import Training


STAGE_NAME = "Training stage"
class TrainingPipeline:
    def __init__(self):
        pass 

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.train()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>> The stage {STAGE_NAME} has started >>>>>>>>>>")
        training = TrainingPipeline()
        training.main()
        logger.info(f">>>>>>>>>The stage {STAGE_NAME} has completed successfulyy >>>>>>>>>>>>")
    except Exception as e:
        logger.exception(e)
        raise e
