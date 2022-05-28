Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# ML Pipeline deployment API on Heroku
In this project an ML pipeline was deployed and exposed on Heroku.
Here the pipeline was trained and published on Heroku

## Environment
### Githooks
* Install Flake8 githooks on local development. See steps below:
* `pip install pre-commit`

### Testing the application
The application can be tested using Pytest.

The test was designed to test the core functions of the program.

### Functions created
1. Data preprocessing. To use: `python main.py --action basic cleaning`
2. Train/test model. To use: `python main.py --action train_test_model`
3. Score slices. To use: `python main.py --action check_score`

### Running the full pipeline
1. To run the entire pipeline: `python main.py --action all`
2. To serve the API on local: `uvicorn api_server:app --reload`
3. Use the Heroku API: `python check_heroku_api.py`

### CI/CD
For the CI/CD Heroku was used. This repository was connected to Heroku.
Every section of the pipeline was executed upon a pull request. The test pipeline is first triggered.
The Pipeline pulls data from DVC and execute Flake8 + pytest executing the test.

