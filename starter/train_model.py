### 
# Script to train machine learning model.
###

import joblib
import os

import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Get the current working directory
cwd = os.getcwd()
print("Current working directory:", cwd)

# Move up to the parent directory
parent_dir = os.path.dirname(cwd)
print("Parent directory:", parent_dir)

# Name of the file you want to locate
filename = "data\census.csv"

# Construct the full path to the file in the parent directory
file_path = os.path.join(parent_dir, filename)
print("Full path to the file:", file_path)
# Add code to load in the data.
data = pd.read_csv(file_path, sep=", ", engine="python")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=101)

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
    train, 
    categorical_features=cat_features, 
    label="salary", 
    training=True
)

# Proces the test data with the process_data function.

X_test, y_test, * _ = process_data(test, 
                                   categorical_features=cat_features,
                                   label="salary",
                                   training=False,
                                   encoder=encoder,
                                   lb=lb)

# Train and save a model.
model = train_model(X_train=X_train, y_train=y_train)

print(f"Best Parameters: {model.best_params_}")

y_pred = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

results = {"precision": precision,
           "recall":recall,
           "fbeta":fbeta}

print(results)

joblib.dump((model, encoder, lb),
            "./model/transformers.pkl")

with open("./results.txt", "w+") as file:
    text = str(model.best_params_) + "\n" + str(results)
    file.write(text)