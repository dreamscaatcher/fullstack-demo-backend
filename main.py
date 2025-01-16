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

model = joblib.load('lin_regress.sav')

@app.post("/predict")
async def predict(features: PredictionFeatures):
    try:
        input_df = pd.DataFrame([{
            'experience_level': features.experience_level,
            'company_size': features.company_size,
            'employment_type': features.employment_type,
            'job_title': features.job_title
        }])

        encoder = OrdinalEncoder(categories=[['EN', 'MI', 'SE', 'EX']])
        input_df['experience_level_encoded'] = encoder.fit_transform(input_df[['experience_level']])

        encoder = OrdinalEncoder(categories=[['S', 'M', 'L']])
        input_df['company_size_encoded'] = encoder.fit_transform(input_df[['company_size']])

        input_df = pd.get_dummies(input_df, columns=['employment_type', 'job_title'])
        
        # Get expected column names from model
        logger.debug(f"Input columns: {input_df.columns}")
        
        required_columns = ['experience_level_encoded', 'company_size_encoded', 
                          'employment_type_PT', 'job_title_Data Manager',
                          'job_title_Data Scientist', 'job_title_Machine Learning Engineer']
                          
        # Initialize missing columns with 0
        for col in required_columns:
            if col not in input_df.columns:
                input_df[col] = 0
                
        input_df = input_df[required_columns]
        
        prediction = model.predict(input_df)[0]
        return {"salary_prediction_usd": round(prediction, 2)}
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))