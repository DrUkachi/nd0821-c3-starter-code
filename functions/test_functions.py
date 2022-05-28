"""
This is going to be used to test all the functions created in:
    1. model.py
    2. data.py
"""

import pytest
import pandas as pd
import logging
import model


@pytest.fixture
def get_processed_data():
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data = pd.read_csv("cleaned_data.csv")
    return data.processed_data(data,
                               cat_features,
                               label=None,
                               training=True,
                               encoder=None,
                               lb=None)


@pytest.fixture
def get_trained_model():
    return model.train_model


def test_process_data(get_processed_data):
    X, y, encoder, lb = get_processed_data

    try:
        assert X.shape > 10000
        assert len(y) > 10000

    except AssertionError as err:
        logging.info("Data not processed properly")
        raise err


def test_inference(get_processed_data):
    X,_ ,_ , _ = get_processed_data
    model_trained = get_trained_model

    y_preds = model.inference(model_trained, X)

    assert len(y_preds) > 0


def test_compute_model_metrics(get_processed_data):
    X, y, _, _ = get_processed_data
    model_trained = get_trained_model
    y_preds = model.inference(model_trained,X)
    precision, recall, fbeta = model.compute_model_metrics(y, y_preds)
    assert precision > 0.5
    assert recall > 0.5
    assert fbeta > 0.5
