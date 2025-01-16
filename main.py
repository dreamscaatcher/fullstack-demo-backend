from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Salary Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionFeatures(BaseModel):
    experience_level: str
    company_size: str
    employment_type: str
    job_title: str

    class Config:
        schema_extra = {
            "example": {
                "experience_level": "EN",
                "company_size": "S",
                "employment_type": "FT",
                "job_title": "Data Engineer"
            }
        }

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Salary Prediction API"}

@app.post("/predict")
async def predict(features: PredictionFeatures):
    logger.debug(f"Received features: {features}")
    return {"salary_prediction_usd": 100000}