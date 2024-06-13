import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from starter.ml.model import inference, compute_model_metrics
from starter.ml.data import process_data

# Define categorical features
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

# Load data
data = pd.read_csv("./data/census.csv", sep=", ", engine='python')

# Split the data into training and test sets
_, test = train_test_split(data, test_size=0.20, random_state=42)

# Load model and encoders
model, encoder, lb = joblib.load("./model/transformers.pkl")

# Process test data
X_test, y_test, *_ = process_data(
    test, 
    categorical_features=cat_features, 
    label="salary", 
    training=False, 
    encoder=encoder, 
    lb=lb
)

# Perform inference
y_pred = inference(model, X_test)

# Initialize the results list
list_res = []

# Compute model metrics for each feature and its unique values
for feature in cat_features:
    unique_vals = test[feature].unique()
    for val in unique_vals:
        mask = test[feature] == val
        if mask.any():
            precision, recall, fbeta = compute_model_metrics(y_test[mask], y_pred[mask])
            list_res.append({
                "feature": feature,
                "val": val,
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta
            })

# Create DataFrame and save results to CSV
slices = pd.DataFrame(list_res)
slices.to_csv("./slice_output.txt", index=False)

# Print a sample of the results
print(slices.sample(15))
