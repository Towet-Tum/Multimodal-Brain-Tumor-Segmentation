import os
from django.shortcuts import render
from tumorsegmentation.pipeline.stage_05_predict import PredictionPipeline
import keras
from .forms import PredictForm
import numpy as np
from keras.models import load_model
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image



def to_data_uri(img):
        data = BytesIO()
        img.save(data, "JPEG") # pick your format
        data64 = base64.b64encode(data.getvalue())
        return u'data:img/jpeg;base64,'+data64.decode('utf-8') 

# Create your views here.
def predict(request):
    if request.method == "POST" and request.FILES['pic']:
       
        
        input_image_path = request.FILES['pic']
        input_image = np.load(input_image_path)
        pred = PredictionPipeline(input_image)
        images = pred.make_prediction()

        return render(request, 'results.html', {'images': images})
    
    return render(request, 'results.html')