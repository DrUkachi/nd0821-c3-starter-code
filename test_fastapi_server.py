"""
This module is used to carry out unit testing on the fastapi_server.py script.
"""
import pytest
from fastapi.testclient import TestClient
from api_server import app


@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client


def test_get(client):
    req = client.get("/")
    assert req.status_code == 200
    assert req.json() == {"message": "Greetings!"}


def test_get_malformed(client):
    req = client.get("/wrong_url")
    assert req.status_code != 200


def test_post_above(client):
    req = client.post("/", json={
        "age": 32,
        "workclass": "Private",
        "education": "Some-college",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 60,
        "nativeCountry": "United-States"
    })
    assert req.status_code == 200
    assert req.json() == {"prediction": ">50K"}


def test_post_below(client):
    req = client.post("/", json={
        "age": 19,
        "workclass": "Private",
        "education": "HS-grad",
        "maritalStatus": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
    })
    assert req.status_code == 200
    assert req.json() == {"prediction": "<=50K"}


def test_post_malformed(client):
    req = client.post("/", json={
        "age": 32,
        "workclass": "Private",
        "education": "Some-college",
        "maritalStatus": "ERROR",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 60,
        "nativeCountry": "United-States"
    })
    assert req.status_code == 422