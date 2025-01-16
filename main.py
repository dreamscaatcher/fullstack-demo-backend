from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Salary Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fullstack-demo-frontend-production.up.railway.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionFeatures(BaseModel):
    experience_level: str
    company_size: str
    employment_type: str
    job_title: str

# Load model
model = joblib.load('lin_regress.sav')

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Salary Prediction API"}

@app.post("/predict")
async def predict(features: PredictionFeatures):
    try:
        # Create DataFrame with single row
        input_df = pd.DataFrame([{
            'experience_level': features.experience_level,
            'company_size': features.company_size,
            'employment_type': features.employment_type,
            'job_title': features.job_title
        }])

        # Encode experience level
        encoder = OrdinalEncoder(categories=[['EN', 'MI', 'SE', 'EX']])
        input_df['experience_level_encoded'] = encoder.fit_transform(input_df[['experience_level']])

        # Encode company size
        encoder = OrdinalEncoder(categories=[['S', 'M', 'L']])
        input_df['company_size_encoded'] = encoder.fit_transform(input_df[['company_size']])

        # Create dummies for employment type and job title
        input_df = pd.get_dummies(input_df, columns=['employment_type', 'job_title'], drop_first=True)

        # Drop original columns
        input_df = input_df.drop(columns=['experience_level', 'company_size'])

        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return {
            "salary_prediction_usd": round(prediction, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)