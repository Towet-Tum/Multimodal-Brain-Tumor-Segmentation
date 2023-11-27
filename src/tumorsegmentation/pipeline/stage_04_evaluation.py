from tumorsegmentation import logger 
from tumorsegmentation.config.configuration import ConfigurationManager 
from tumorsegmentation.components.evalutation import Evaluation
STAGE_NAME = "Evaluation Stage"
class EvaluationPipeline:
    def __init__(self):
        pass 

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()



if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> The stage {STAGE_NAME} has started >>>>>>>>>>>>")
        evaluations = EvaluationPipeline()
        evaluations.main()
        logger.info(f">>>>>>>>>>>>>> The stage {STAGE_NAME} has completed successfully >>>>>>>>>>>")
    except Exception as e:
        logger.exception(e)
        raise e