# Script to train machine learning model.
import pandas as pd
from functions.data import process_data
from functions.model import train_model
from sklearn.model_selection import train_test_split
import joblib


# Add the necessary imports for the functions code.
def train_test():
    # Add code to load in the data.
    data = pd.read_csv("data/cleaned_data.csv")
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

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
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False
    )
    # Train and save a model.
    model = train_model(X_train, y_train)
    joblib.dump(model, "data/model/model.joblib")
    joblib.dump(encoder, "model/encoder.joblib")
    joblib.dump(lb, "model/lb.joblib")
