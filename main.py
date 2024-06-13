# Put the code for your API here.
import joblib
import os
import pandas as pd
from fastapi import FastAPI
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

    model_config = {"json_schema": {
        "examples": [
            {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-Family",
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
    }}

# Initiate the app
app = FastAPI()


# Define a GET method on the specified end-point
@app.get("/")
async def hello():
    return "Welcome! This is the Census Bureau Classifier API"


cwd = os.getcwd()
parent_dir = os.path.dirname("model")

# Name of the file you want to locate
filename = "transformers.pkl"

# Construct the full path to the file in the parent directory
file_path = os.path.join(parent_dir, filename)
print(file_path)

# Load model and transformers for testing
model, encoder, lb = joblib.load(file_path)

cat_features = [f for (f, t) in Features.__annotations__.items() if t == str]


# Use Post action to send data to the API
@app.post("/predict")
async def predict(body: Features):
    data = pd.DataFrame(body.__dict__, [0])
    data, *_ = process_data(
        data, categorical_features=cat_features,
        training=False,
        encoder=encoder
    )
    y_pred = inference(model, data)

    return lb.inverse_transform(y_pred)[0]
