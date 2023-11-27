import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import base64
from io import BytesIO


class PredictionPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
   
    
    def to_data_uri(self, img):
        data = BytesIO()
        img.save(data, "JPEG") # pick your format
        data64 = base64.b64encode(data.getvalue())
        return u'data:img/jpeg;base64,'+data64.decode('utf-8') 
    
    def make_prediction(self):
        model = load_model('/home/towet/Desktop/Visions/tumors/Multimodal-Brain-Tumor-Segmentation/artifacts/training/brats_2020.hdf5',
                      compile=False)
        
        test_img_input = np.expand_dims(self.file_path, axis=0)
        test_prediction = model.predict(test_img_input)
        test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]
        num_classes = np.unique(test_prediction_argmax)
        images = []
        predictions = []
        for i in range(test_prediction_argmax.shape[2]):
            #print(test_prediction_argmax[i].shape)
            predictions.append(test_prediction_argmax[:,:,i])

        
        for mask in predictions:
            #print(i.shape)
            # Assuming you have four classes
            color_unlabeled = [0, 0, 0]    # Black
            color_necrotic = [255, 0, 0]   # Red
            color_edema = [0, 255, 0]      # Green
            color_enhancing = [0, 0, 255]  # Blue

            class_to_color = {
                0: color_unlabeled,
                1: color_necrotic,
                2: color_edema,
                3: color_enhancing
            }
            # Assuming you have intensity values corresponding to each class
            class_to_intensity = {
                0: 0,
                1: 1,
                2: 2,
                3: 3
            }   
                    
            
            larger_colored_mask = np.zeros((500, 500, 3), dtype=np.uint8)
            # Map class values to colors
            # Map class values to intensities
            for class_value, intensity in class_to_intensity.items():
                larger_colored_mask[np.where(mask == intensity)] = class_to_color[class_value]

            # Create a PIL Image from the numpy array
            larger_image = Image.fromarray(larger_colored_mask)
            
            img_uri = self.to_data_uri(larger_image)
            images.append(img_uri)
            print(num_classes)

        """"
        images = []
            for i in range(128):
                    img = self.to_image(test_prediction_argmax[:,:,i])
                    image_uri = self.to_data_uri(img)
                    images.append(image_uri)
        """
            
        return images
       
        
       

    """
            def preprocess_image(self, image_path):
            img = np.load(image_path)
            #img_np = np.array(img.dataobj)
            img_arr = np.expand_dims(img, axis=0)
            return img_arr
        
        def predict_segmentation(self, input_image):
            segmentation_mask = self.model.predict(input_image)
            segmentation_mask_argmax = np.argmax(segmentation_mask, axis=4)[0,:,:,:]
            return segmentation_mask_argmax
        

        def mask_to_image(self, segmentation_mask_armax):
            predicted_class = np.argmax(segmentation_mask_armax, axis=-1)[0]
            colormap = plt.get_cmap('viridis')
            colored_mask = colormap(predicted_class)
            colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)
            return Image.fromarray(colored_mask)
    """

if __name__ == "__main__":
    test_input = np.load("/home/towet/Desktop/Visions/tumors/Multimodal-Brain-Tumor-Segmentation/artifacts/preprocessed_data/train_val_dataset/val/images/image_41.npy")
    predictions = PredictionPipeline(test_input)
    predictions.make_prediction()

