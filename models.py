
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
    
class PriceCrop(Enum):
    MANGO="mango"
    BANANA="banana"
    BLACK_PEPPER = "black pepper"
    BRINJAL= "brinjal"
    CARROT="carrot"
    GREEN_CHILLI = "green chilli"
    GREEN_PEAS = "green peas"
    JACK_FRUIT="jack fruit"
    ONION = "onion"
    PAPAYA= "papaya"
    RICE = "rice"
    SWEET_POTATO = "sweet potato"
    TAPIOCA = "tapioca"


class PredictPriceRequest(BaseModel):
    crop: PriceCrop
    steps: int


