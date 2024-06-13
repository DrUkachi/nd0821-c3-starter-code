import requests
import json

def send_prediction_request():
    """
    Sends a POST request to a web API endpoint with sample data and prints the response.
    
    Returns
    -------
    None
    """
    # Sample data to send as JSON
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

    # Convert data to JSON format
    json_data = json.dumps(data)

    # Send POST request to API endpoint
    url = "https://census-web-app.onrender.com/predict"
    response = requests.post(url, data=json_data)

    # Print status code and result
    print("status_code:", response.status_code)
    print("result:", response.json())

# Call the function to send the request
send_prediction_request()
