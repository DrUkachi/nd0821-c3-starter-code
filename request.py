import requests
import json

# Define the URL
url = 'https://nd0821-c3-starter-code-1-ccfs.onrender.com/predict'

# Define the headers
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

# Define the data payload
data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(data))

# Print the response
print(response.status_code)
print(response.json())
