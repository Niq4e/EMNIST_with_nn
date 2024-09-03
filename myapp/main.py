import torch
from torch import nn
import numpy as np


from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from myapp.model import Model

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



# load model
model = Model()

# app
app = FastAPI(title='Symbol detection', docs_url='/docs')

# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):
    #torch.save(image, 'image0.pkl')
    image = torch.tensor(list(map(int, image[1:-1].split(',')))).reshape((28, 28))
    #torch.save(image1, 'image.pkl')
    #image2 = np.array(list(map(int, image[1:-1].split(',')))).reshape((28, 28))
    #torch.save(image2, 'image2.pkl')
    pred = model.predict(image)
    return {'prediction': pred}

# static files
app.mount('/', StaticFiles(directory='static', html=True), name='static')
