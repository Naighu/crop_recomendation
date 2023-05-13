
from typing import List, Union
from pydantic import BaseModel,Field
from enum import Enum

class Model(str,Enum):
    MLP = 'mlp'
    SVC = 'svc'
    RFC = 'rfc'
    GNB  = 'gnb'

class PredictCrop(BaseModel):
    limit: int = Field(
        alias="limit",
        default=5
    )
    model: Model = Field(
        alias="model"
    )
    nitrogen: float = Field(
	    alias="nitrogen",
       
    )
    phosphorus:  float = Field(
	    alias="phosphorus",
       
    )
    potassium:  float = Field(
	    alias="potassium",
       
    )
    temperature: float = Field(
	    alias="temperature",
       
    ),
    humidity: float = Field(
	    alias="humidity",
       
    ),
    ph: float = Field(
	    alias="ph",
       
    ),
    rainfall: float = Field(
	    alias="rainfall",
       
    ),
    