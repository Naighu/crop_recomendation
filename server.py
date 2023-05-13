from fastapi import FastAPI,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from models import PredictCrop,Model,PriceCrop,PredictPrice
import logging,os
import numpy as np
import crop_rec as crop
import uvicorn

from price_pred import predict_price

app = FastAPI()

logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error.log'), filemode='a', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)


def get_model_from_str(model: Model):

    if(model == Model.MLP):
        return crop.mlp

    if model == Model.GNB:
        return crop.gnb

    if model == Model.RFC:
        return crop.rfc
    
    if model == Model.SVC:
        return crop.svc
    
def get_price_model_file(crop: PriceCrop):

    if(crop == PriceCrop.MANGO):
        return "prediction_models/mango.pkl"
    


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=200,
      content=  {"response_code": 400,"message" : "Data validation error"})


@app.post("/recommend_crop",status_code=201)
async def root(cropPredict: PredictCrop):
    try:
        
        input = np.array([[cropPredict.nitrogen,cropPredict.phosphorus,cropPredict.potassium,cropPredict.temperature,cropPredict.humidity,cropPredict.ph,cropPredict.rainfall]])
        
        model = get_model_from_str(cropPredict.model)
        if not model:
            return  {"response_code": 400,"message" : "Invalid model"}
        
        results  = crop.predict(model,input)



        if len(results) >= cropPredict.limit:
            results = results[:cropPredict.limit]
        return {"response_code:": 200,"message": "Crop recommended","response" : {
            "crops" : results
    
        }}

    except Exception as e:
        logger.error(e)
        print(e)
        return {"response_code": 500,"message" : "Something went wrong", "error": e}

@app.post("/predict_price",status_code=201)
async def root(priceRequest: PredictPrice):
    try:

        model_path = get_price_model_file(priceRequest.crop)
        if not model_path:
            return {"response_code:": 400,"message": "Invalid crop"}

        results= predict_price(model_path,priceRequest.steps)
        
        return {"response_code:": 200,"message": "Crop recommended","response" : {
            "price" : results
    
        }}

    except Exception as e:
        logger.error(e)
        print(e)
        return {"response_code": 500,"message" : "Something went wrong", "error": e}


uvicorn.run(app, port=6666)