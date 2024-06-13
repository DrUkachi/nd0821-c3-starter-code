import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from starter.ml.model import train_model, compute_model_metrics, inference

def test_train_model():
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, 
                               n_features=10, 
                               random_state=42)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Validate the model
    assert hasattr(model, 
        'best_estimator_'), "GridSearchCV should have best_estimator_ attribute"
    assert model.best_score_ > 0, "GridSearchCV should have non-zero best_score_"


def test_compute_model_metrics():
    # Simulate true labels and predicted labels
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 1])

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Validate the metrics
    assert precision >= 0 and precision <= 1, "Precision score should be between 0 and 1"
    assert recall >= 0 and recall <= 1, "Recall score should be between 0 and 1"
    assert fbeta >= 0 and fbeta <= 1, "F-beta score should be between 0 and 1"

def test_inference():
    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    # Train a model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    # Perform inference
    y_pred = inference(model, X)

    # Validate the shape of predictions
    assert y_pred.shape[0] == X.shape[0], "Number of predictions should match number of samples"

    # Validate the type of predictions (assuming binary classification)
    assert np.all(np.isin(y_pred, [0, 1])), "Predictions should be binary (0 or 1)"
