"""
This module is used to consume/test the Heroku API
"""
import requests

data = {
    "age": 32,
    "workclass": "Private",
    "education": "Some-college",
    "maritalStatus": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "Black",
    "sex": "Male",
    "hoursPerWeek": 60,
    "nativeCountry": "United-States"
}

req = requests.post('https://ml-heroku-fastapi.herokuapp.com/', json=data)

assert req.status_code == 200

print(f"Response code: {req.status_code}")
print(f"Response body: {req.json()}")
