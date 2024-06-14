import pandas as pd

# Load the data once for all tests to use
data = pd.read_csv("./data/census.csv", sep=",", engine='python')

def test_required_columns():
    required_columns = [
        "age", "workclass", "education", "marital-status", "occupation", 
        "relationship", "race", "sex", "native-country", "salary"
    ]
    assert all(column in data.columns for column in required_columns), "Missing required columns"

def test_data_types():
    assert data['age'].dtype == 'int64', "Incorrect data type for age"
    assert data['workclass'].dtype == 'object', "Incorrect data type for workclass"
    assert data['education'].dtype == 'object', "Incorrect data type for education"
    assert data['marital-status'].dtype == 'object', "Incorrect data type for marital-status"
    assert data['occupation'].dtype == 'object', "Incorrect data type for occupation"
    assert data['relationship'].dtype == 'object', "Incorrect data type for relationship"
    assert data['race'].dtype == 'object', "Incorrect data type for race"
    assert data['sex'].dtype == 'object', "Incorrect data type for sex"
    assert data['native-country'].dtype == 'object', "Incorrect data type for native-country"
    assert data['salary'].dtype == 'object', "Incorrect data type for salary"

def test_no_missing_values():
    assert data.isnull().sum().sum() == 0, "There are missing values in the dataset"

def test_age_range():
    assert data['age'].between(0, 100).all(), "Age values out of range"

def test_unique_education_levels():
    unique_education_levels = data['education'].unique()
    expected_education_levels = [
        "Bachelors", "Masters", "Doctorate", "Some-college", "Assoc-acdm", 
        "Assoc-voc", "HS-grad", "12th", "11th", "10th", "9th", "7th-8th", 
        "5th-6th", "1st-4th", "Preschool"
    ]
    assert set(unique_education_levels) == set(expected_education_levels), "Unexpected education levels"

def test_valid_workclass():
    valid_workclass = [
        "State-gov", "Self-emp-not-inc", "Private", "Federal-gov", 
        "Local-gov", "Self-emp-inc", "Without-pay", "Never-worked", "Unknown"
    ]
    assert all(workclass in valid_workclass for workclass in data['workclass'].unique()), "Invalid workclass value found"
