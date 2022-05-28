# Model Card


## Model Details
(Basic information about the model)

* Developed by Ukachi Osisiogu
* Date: 28th May 2022
* Model Version: 1.0.0
* Model Type: Gradient Boosting Classifier

## Intended Use
(Use cases that were envisioned during development)
* The model is intended to predict the salary of a person based on census data.
* Users: Governments, businesses, marketing, research

## Factors
(What factors affect the impact of the model?)
* Several categories
* Model was trained based on several categorical slices and generated different F1 scores.

## Metrics
(What metrics are you using to measure the performance of the model?)

* F1 score
* AUC score
* ROC score

## Evaluation of Data
(Details on the dataset(s) used for the
quantitative analyses in the card.)

Information about the data can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income)

The train data was made up of 80% of the cleaned data. Of that 80%, 80% was used for training 
while the remaining 20 % was used for validation.

The remaining 20% (of the cleaned data)  was then used as the test dataset.

## Ethical Considerations
The dataset contains information related to gender, age, race etc.
This data may be prone discrimination this further scrutiny is advised to prevent bias.

## Caveats and Recommendations
Given the population of male and female in the data is unbalanced
proper investigation/concern is needed to ensure the ML pipeline for future work
enforces balance. This would ensure ethical considerations are not left out.