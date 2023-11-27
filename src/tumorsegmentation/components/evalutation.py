from keras.models import load_model
from urllib.parse import urlparse
from tumorsegmentation.utils.common import imageLoader
import os
from keras.metrics import MeanIoU
import numpy as np
from tumorsegmentation.entity.config_entity import EvaluationConfig
from pathlib import Path
from tumorsegmentation.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
  
        
    
    def evaluation(self):


        val_img_dir = self.config.val_img_dir
        val_mask_dir = self.config.val_mask_dir
        val_img_list = os.listdir(val_img_dir)
        val_mask_list = os.listdir(val_mask_dir)
        
        test_img_datagen = imageLoader(val_img_dir, val_img_list,
                                val_mask_dir, val_mask_list, self.config.batch_size)


        #Verify generator.... In python 3 next() is renamed as __next__()
        test_image_batch, test_mask_batch = test_img_datagen.__next__()

        test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
        model = load_model(self.config.path_of_model, compile=False)
        test_pred_batch = model.predict(test_image_batch)
        test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

       
        IOU_keras = MeanIoU(num_classes=self.config.num_classes)
        IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
        #print("Mean IoU =", IOU_keras.result().numpy())
        scores = {"Mean IOU" : IOU_keras.result().numpy()}
        save_json(path=Path("scores.json"), data=scores)