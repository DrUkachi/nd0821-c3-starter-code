# Deploying a Machine Learning Model with FastAPI

## Project Overview
This project demonstrates how to deploy a machine learning model using FastAPI. The project includes training a model, creating an API for model inference, and deploying the API to a cloud application platform with continuous integration and delivery.

## Table of Contents
- [Project Overview](#project-overview)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Endpoints](#api-endpoints)
- [Continuous Integration](#continuous-integration)
- [Deployment](#deployment)
- [Testing](#testing)
- [Model Card](#model-card)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Installation
1. Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Train the model:
    ```bash
    python train_model.py
    ```

2. Run the API:
    ```bash
    uvicorn main:app --reload
    ```

## Model Training
The model is trained using the provided dataset. The training script processes the data, trains the model, and saves the model and encoders.

## API Endpoints
- **GET /**: Returns a greeting message.
- **POST /predict**: Takes input data and returns model inference.

## Continuous Integration
Continuous integration is set up using GitHub Actions. The CI runs `pytest` and `flake8` on push to the main branch.

![Continuous Integration](<https://github.com/DrUkachi/nd0821-c3-starter-code/blob/main/screenshots/continuous_integration.png>)

## Deployment
The API is deployed using a cloud application platform with continuous delivery enabled.

![Continuous Deployment](<https://github.com/DrUkachi/nd0821-c3-starter-code/blob/main/screenshots/continuous_deloyment.png>)

![Live GET Request](<https://github.com/DrUkachi/nd0821-c3-starter-code/blob/main/screenshots/live_get.png>)

![Screenshot of Live Swagger UI Docs on the deplyed Server](<https://github.com/DrUkachi/nd0821-c3-starter-code/blob/main/screenshots/example.png>)

## Testing
The project includes unit tests for the model and API tests for the endpoints.

## Model Card
The model card provides details about the model, including performance metrics.

![Model Card](<https://github.com/DrUkachi/nd0821-c3-starter-code/blob/main/model_card.md>)

## Results
- **Slice Output**: Performance metrics for different slices of the data.

    ![Slice Output](<https://github.com/DrUkachi/nd0821-c3-starter-code/blob/main/slice_output.txt>)

- **Live POST Request**: Result of a POST request to the live API.

    ![Live POST Request](<https://github.com/DrUkachi/nd0821-c3-starter-code/blob/main/screenshots/live-post.png>)

## Acknowledgements
This project is part of the Udacity Machine Learning DevOps Engineer Nanodegree.