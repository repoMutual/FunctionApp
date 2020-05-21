import logging
import azure.functions as func
import keras
from keras.models import load_model
import numpy as np


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    img = './covidman.png'
    model = load_model('/covidmodel')
    pred_proba=model.predict(img)
    max_proba=np.argmax(pred_proba,axis=1)
    return max_proba	
#     class prediction:
# def predict(img):

#img='D:/Covid19Detection/data/covid/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg'
    # name = req.params.get('name')
    # if not name:
    #     try:
    #         req_body = req.get_json()
    #     except ValueError:
    #         pass
    #     else:
    #         name = req_body.get('name')

    # if name:
    #     return func.HttpResponse(f"Hello {name}!")
    # else:
    #     return func.HttpResponse(
    #          "Please pass a name on the query string or in the request body",
    #          status_code=400
    #     )
