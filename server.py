from fastapi import FastAPI,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from models import PredictCrop,Model
import logging,os
import numpy as np
import crop_rec as crop
import uvicorn

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


uvicorn.run(app, port=6666)