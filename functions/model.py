import logging

import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

import functions.data as data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy',
                             cv=cv, n_jobs=-1)
    logging.info('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_preds = model.predict(X)
    return y_preds


def score_slices():
    """
    This function is responsible for slicing the data (using different categories) and scoring the model
    Returns
    -------

    """
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

    data_df = pd.read("cleaned_data.csv")
    train, test = train_test_split(data_df, test_size=0.2)

    trained_model = load("functions/model/model.joblib")
    one_hot_encoder = load("functions/model/encoder.joblib")
    label_encoder = load("functions/model/lb.joblib")

    slice_scores = []

    for category in cat_features:
        for sample in test[category].unique():
            sample_df = test[test[category] == sample]

            X_test, y_test, _, _ = data.process_data(sample_df,
                                                     categorical_features=cat_features,
                                                     label="salary",
                                                     encoder=one_hot_encoder,
                                                     lb=label_encoder,
                                                     training=False)

            y_pred = trained_model.predict(X_test)

            precision_score, recall_score, fbeta_score = compute_model_metrics(y_test,
                                                                               y_pred)
            line = f"""[{category}=>{sample}] - Precision: {precision_score} 
            Recall: {recall_score}, fbeta: {fbeta_score}"""
            logging.info(line)
            slice_scores.append(line)

    with open("functions/functions/ml/slice_score_output.txt") as output_file:
        for slice_score in slice_scores:
            output_file.write(slice_score + "\n")
