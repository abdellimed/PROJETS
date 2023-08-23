import json
import os

import warnings
from typing import List

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import string
import nltk

app = FastAPI()

class Commentaire(BaseModel):
    Comment: str
@app.get("/")
def root():
    return {"API pour classification des commentaires" }

@app.post("/Comment-class")
def post_Comment_class(clsT: Commentaire):
    # os.environ['AWS_ACCESS_KEY_ID']="mlflow_access"
    #  os.environ['AWS_DEFAULT_REGION']="us-east-1"
    #  os.environ['MLFLOW_S3_ENDPOINT_URL']="http://10.185.33.168:9000"
    #  os.environ['AWS_SECRET_ACCESS_KEY']="mlflow_secret"
    remote_server_uri =os.environ.get('MLFLOW_TRACKING_URI')
    #  "http://10.185.33.168:5000"
    mlflow.set_tracking_uri(remote_server_uri)
    model_uri ="models:/classification/Production"
    model=mlflow.pyfunc.load_model(model_uri)
    res = model.predict(clsT.Comment)[0]
    return {res}


@app.post("/Comments-classes")
def get_comment_classes(comments: List[Commentaire]):
  # set to your server URI
    remote_server_uri =os.environ.get('MLFLOW_TRACKING_URI')

    mlflow.set_tracking_uri(remote_server_uri)

    model_uri ="models:/classification/Production"

    model=mlflow.pyfunc.load_model(model_uri)

    results = []
    for cmnt in comments:
        results.append(model.predict(cmnt.Comment)[0])


    return {"predictions": results}
