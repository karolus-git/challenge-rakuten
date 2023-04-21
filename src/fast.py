
from fastapi import FastAPI, Request, status
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, BaseSettings

import importlib
import sys
import torch
import PIL
import numpy as np
import requests
from pathlib import Path
from hydra import compose, initialize
from omegaconf import OmegaConf

import models
from utils import preprocessors
from utils import explainers
from utils import predictors
from train import run

#Where to pu your image from url ?
UPLOAD_FOLDER = "./temp"
UPLOAD_FILENAME = "uploaded_image.jpg"

#Information about the model
class ModelInfo(BaseModel):
    name: str
    version: str
    ckpt: str

#Differents models to load
class Models(BaseModel):
    image: ModelInfo
    text: ModelInfo
    fusion: ModelInfo

#Settings
class Settings(BaseSettings):
    model: Models

    class Config:
        env_file = ".env.fast"
        env_nested_delimiter = '__'

#Fields for text_input
class TextInput(BaseModel):
    text: str = Field(..., example="Ceci est une piscine tubulaire de grand volume pleine d'eau", title='Text [-]')
    evals: int = Field(..., example=100)
    topk: int = Field(..., example=5)

#Fields for image inputs
class ImageInput(BaseModel):
    image_url: str = Field(..., example="https://fr.shopping.rakuten.com/photo/2444779481.jpg", title='URL [-]')
    evals: int = Field(..., example=100)
    topk: int = Field(..., example=5)

#Fields for image inputs
class FusionInput(BaseModel):
    text: str = Field(..., example="Ceci est une piscine tubulaire de grand volume pleine d'eau", title='Text [-]')
    image_url: str = Field(..., example="https://fr.shopping.rakuten.com/photo/2444779481.jpg", title='URL [-]')
    evals: int = Field(..., example=100)
    topk: int = Field(..., example=5)

def get_ckpt_path(name, version, ckpt):

    return f"./runs/{name}/{version}/checkpoints/{ckpt}.ckpt"

def dynamic_import(name):
    return importlib.import_module(f"models.{name}")

def load_cfg(overrides=[]):
    
    initialize(config_path="conf", job_name="config")
    cfg = compose(config_name="config", overrides=overrides)

    return cfg

def load_model(settings):
    #Get settings
    model_name = settings.name
    model_version = settings.version
    model_ckpt = settings.ckpt
    
    #Get path of checkpoint
    model_ckpt_path = get_ckpt_path(model_name, model_version, model_ckpt)
    
    #Load the right module
    module = dynamic_import(model_name)

    #From this module, get the model
    model = getattr(module, model_name)

    #Load the weights
    model = model.load_from_checkpoint(model_ckpt_path)
    model.eval()

    print("imported : ",model_name)
    return model

def save_image(url):
    #Get the image
    img_data = requests.get(str(url)).content

    #Create path
    path = Path(UPLOAD_FOLDER, UPLOAD_FILENAME)
    if not path.parent.exists():
        path.parent.mkdir(exist_ok=True, parents=True)

    #Write it to path
    with open(path, 'wb') as handler:
        handler.write(img_data)

# Initialize API Server
app = FastAPI(
    title="Challenge Rakuten Models",
    description="Challenge Rakuten",
    version="0.0.1",
)
settings = Settings()
dl_models = {}

@app.get('/')
def show_ping():
    """Am I alive ? """
    return {
        "Yes, I am alive ;)"
    }
    
@app.on_event("startup")
def startup_event():
    """Load the models on startup according to the settings"""
    print("starting")
    
    for name in ["image", "fusion", "text"]:
        model_info = vars(settings.model).get(name)
        dl_models[name] = load_model(model_info)

@app.post('/api/v1/train',)
def train_model(request: Request, body: ModelInfo):
    
    #Get the configuration
    model_to_train = f"model={body.name}"
    cfg = load_cfg(overrides=[model_to_train])

    #Run the training
    run(cfg)

    return {
        "Training done"
    }

@app.post('/api/v1/predict/image',)
def predict_image(request: Request, body: ImageInput):
    """
    Perform prediction on image data
    """

    try:
        #Save the image from URL
        save_image(body.image_url)
        
        #Load it
        image = PIL.Image.open(Path(UPLOAD_FOLDER, UPLOAD_FILENAME))

        #Make prediction
        y_hat_numpy = predictors.image([image,], dl_models["image"])

        return {
            "error": False,
            "results": y_hat_numpy[0].tolist()
        }

    except Exception as exce:

        return {
            "error": exce,
            "results": None
        }

@app.post('/api/v1/predict/text',)
def predict_text(request: Request, body: TextInput):
    """
    Perform prediction on text data
    """

    try:
        #Get the text to predict
        text = body.text

        #Make the prediction
        y_hat_numpy = predictors.text([text,], dl_models["text"])
        
        return {
            "error": False,
            "results": y_hat_numpy[0].tolist()
        }

    except Exception as exce:
 
        return {
            "error": True,
            "results": None
        }

@app.post('/api/v1/predict/fusion',)
def predict_fusion(request: Request, body: FusionInput):
    """
    Perform prediction on text data
    """

    try:
        #Get the text to predict
        text = body.text

        #Save the image from URL
        save_image(body.image_url)

        #Load it
        image = PIL.Image.open(Path(UPLOAD_FOLDER, UPLOAD_FILENAME))

        #Make the prediction
        y_hat_numpy = predictors.fusion([text,], [image,], dl_models["fusion"])

        return {
            "error": False,
            "results": y_hat_numpy[0].tolist()
        }

    except Exception as exce:
 
        return {
            "error": True,
            "results": None
        }

@app.post('/api/v1/explain/text',)
def explain_text(request: Request, body: TextInput):
    """
    Perform prediction on text data
    """

    try:
        #Get the text to explain
        text = body.text
        print(text)
        shape_values = explainers.text(
            [text,], 
            dl_models["text"],
            topk=body.topk,
            n_evals=body.evals)

        return {
            "error": False,
            "results": shape_values
        }

    except Exception as exce:
        print(exce)
        return {
            "error": exce,
            "results": None
        }

@app.post('/api/v1/explain/image',)
def explain_image(request: Request, body: ImageInput):
    """
    Perform prediction on image data
    """

    try:
        #Save the image from URL
        save_image(body.image_url)
        
        #Load it
        image = PIL.Image.open(Path(UPLOAD_FOLDER, UPLOAD_FILENAME))

        #Explain it
        image_html = explainers.image(
            [image,], 
            model_image, 
            n_evals=body.evals,
            topk=body.topk)

        return {
            "error": False,
            "results": image_html
        }

    except Exception as exce:

        return {
            "error": exce,
            "results": None
        }