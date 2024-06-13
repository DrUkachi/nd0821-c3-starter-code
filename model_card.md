# Model Card

## Model Details
The prediction task for this model is to determine whether an individual's income exceeds $50K annually. We utilize a `GradientBoostingClassifier` with optimized hyperparameters in scikit-learn version 0.23.2. Hyperparameter tuning was conducted using `GridSearchCV`. The optimal parameters found are:

- `learning_rate`: 0.05
- `max_depth`: 5
- `min_samples_split`: 5
- `n_estimators`: 500

## Intended Use

This model is designed to predict an individual's income range, which can be helpful for determining eligibility for certain services or facilities, such as loan approval.

## Training Data

The training data used is the census data available from the UCI Machine Learning Repository. Specifically, it is the "adult" income dataset located in the data folder.
<br />
Link: [UCI Census Data](https://archive.ics.uci.edu/ml/datasets/census+income)

## Evaluation Data

The evaluation data was split from the training set, with an 80:20 split ratio.

## Metrics
The metrics used to evaluate model performance are:

- Precision: 0.7901
- Recall: 0.6805
- F-beta score: 0.7312

## Ethical Considerations

The dataset should not be regarded as a fair representation of the overall salary distribution and should not be used to make assumptions about the income levels of specific population groups.

## Caveats and Recommendations

The data was extracted from the 1994 Census database, making it an outdated sample that does not adequately represent the current population. It is recommended to use this dataset primarily for training purposes in machine learning classification or related tasks.