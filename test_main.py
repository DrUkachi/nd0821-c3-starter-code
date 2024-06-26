import os
import pytest
import json
import joblib
from fastapi.testclient import TestClient
from main import app, Features  # Assuming your FastAPI script is in main.py

# Load model and transformers for testing
file_path = os.path.join(os.getcwd(), "model/transformers.pkl")
model, encoder, lb = joblib.load(file_path)
cat_features = [f for (f, t) in Features.__annotations__.items() if t == str]

# Create a TestClient passing the FastAPI app
client = TestClient(app)

def test_root_endpoint():
    # Test the root endpoint "/"
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome! This is the Census Bureau Classifier API"

@pytest.mark.parametrize("example_index, expected_output", [
    (0, "<=50K"),
    (1, ">50K"),
])
def test_post_prediction(example_index, expected_output):
    examples = Features.Config.schema_extra["example"]
    data = json.dumps(examples[example_index])
    response = client.post("/predict", data=data)
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    assert "prediction" in response.json(), "Response JSON does not contain 'prediction'."
    assert response.json()["prediction"] in ["<=50K", ">50K"], f"Expected '<=50K' or '>50K' but got {response.json()['prediction']}"

def test_invalid_input():
    # Test the "/predict" endpoint with invalid input data
    invalid_data = {
        "age": "invalid",
        "workclass": "Self-emp-not-inc",
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
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Expecting Unprocessable Entity status code for invalid input

def test_missing_fields():
    # Test the "/predict" endpoint with missing fields in input data
    missing_data = {
        "age": 39,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=missing_data)
    assert response.status_code == 422  # Expecting Unprocessable Entity status code for missing fields
