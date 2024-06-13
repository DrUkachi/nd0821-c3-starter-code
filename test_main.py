import pytest
import json
import joblib
import pandas as pd
from fastapi.testclient import TestClient
from main import app, Features  # Assuming your FastAPI script is in main.py

# Load model and transformers for testing
model, encoder, lb = joblib.load("./model/transformer.pkl")
cat_features = [f for (f, t) in Features.__annotations__.items() if t == str]

client = TestClient(app)

@pytest.fixture(scope="module")
def setup():
    # Setup function to initialize resources or state for testing
    return model, encoder, lb, cat_features

def test_root_endpoint():
    # Test the root endpoint "/"
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "This is the Census Bureau Classifier API"


@pytest.fixture(scope="module")
def setup():
    # Setup function to initialize resources or state for testing
    # Load necessary resources or configurations
    return Features.model_config["json_schema"]["examples"]

@pytest.mark.parametrize("example_index, expected_output", [
    (0, "<=50K"),
    (1, ">50K"),
])

def test_post_prediction(setup, example_index, expected_output):
    examples = setup

    # Convert example data to JSON string
    data = json.dumps(examples[example_index])

    # Send POST request to endpoint
    response = client.post("/predict", data=data)

    # Assert the response status code
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"

    # Assert the predicted output matches the expected output
    assert response.json() == expected_output, f"Expected {expected_output} but got {response.json()}"



def test_invalid_input(setup):
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

def test_missing_fields(setup):
    # Test the "/predict" endpoint with missing fields in input data
    model, encoder, lb, cat_features = setup

    missing_data = {
        "age": 39,
        "workclass": "State-gov",
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

    response = client.post("/predict", json=missing_data)
    assert response.status_code == 422  # Expecting Unprocessable Entity status code for missing fields