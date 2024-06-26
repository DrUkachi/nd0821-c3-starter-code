import joblib
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starter.ml.data import process_data
from starter.ml.model import inference

class Features(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education_num": 13,
                    "marital_status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States"
                },
                {
                    'age': 50,
                    'workclass': "Private",
                    'fnlgt': 234721,
                    'education': "Doctorate",
                    'education_num': 16,
                    'marital_status': "Separated",
                    'occupation': "Exec-managerial",
                    'relationship': "Not-in-family",
                    'race': "Black",
                    'sex': "Female",
                    'capital_gain': 0,
                    'capital_loss': 0,
                    'hours_per_week': 50,
                    'native_country': "United-States"
                }
            ]
        }

# Initiate the app
app = FastAPI()

# Define a GET method on the specified end-point
@app.get("/")
async def hello():
    return "Welcome! This is the Census Bureau Classifier API"

def load_model_components():
    try:
        # get the base directory
        cwd = os.getcwd()
        # Name of the file you want to locate
        filename = "model/transformers.pkl"
        # Construct the full path to the file in the parent directory
        file_path = os.path.join(cwd, filename)
        # Load model and transformers for testing
        model, encoder, lb = joblib.load(file_path)
        return model, encoder, lb
    except Exception as e:
        raise RuntimeError(f"Failed to load model components: {e}")

# Load the model and transformers
model, encoder, lb = load_model_components()

cat_features = [f for (f, t) in Features.__annotations__.items() if t == str]

# Use Post action to send data to the API
@app.post("/predict")
async def predict(body: Features):
    try:
        data = pd.DataFrame([body.dict()])
        data, *_ = process_data(
            data, categorical_features=cat_features,
            training=False,
            encoder=encoder
        )
        y_pred = inference(model, data)
        return {"prediction": lb.inverse_transform(y_pred)[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
