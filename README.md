<div align="center">

# fraud-detection-artefact

[![Python Version](https://img.shields.io/badge/python-3.10-blue)]()

[![Linting , formatting, imports sorting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory-nl/fraud-detection/blob/main/.pre-commit-config.yaml)
</div>

This repo aims to build and serve a Fraud Detection model within insurance claims. It leverages an XGBoost model, served through a Flask Rest API. A Streamlit interface allows to make the connection between the user and the API by running inference based on the user's input.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Modelling](#modelling)
  - [API](#api)
  - [User Interface](#user-interface)
- [Development and Testing](#development-and-testing)
  - [Testing](#testing)
  - [Linting and Formatting](#linting-and-formatting)
- [Repository Structure](#repository-structure)

## Installation

To install the required packages in a conda environment, run the following command:

```bash
make install
```

A complete list of available Make commands can be found using the following command:

```bash
make help
```

## Usage

### Modelling

The modelling part mainly consists of the following steps:
1. **Data preprocessing**: This steps consists in mapping the claim identifiers between the two datasets (namely _data/FRISS_ClaimHistory_training.csv_ and _fraud_cases.csv_) to create the training labels.
2. **Feature Engineering**: This step consists in dropping useless columns, as well as sensitive columns, process the date columns, encode the categorical features, and run type conversion.
3. **Model initialization**: Initialization of the FraudDetectionModel object.
4. **Hyperparameters tuning**: Choose the best HP combination, based on a stratified K-fold validation, maximizing the ROC-AUC, through the Bayesian Optimization framework [Optuna](https://optuna.org/).
5. **Threshold definition for predictions**: Choose the best probability threshold to maximize the F1-score on a hold out part of the training set.
6. **Evaluation**: Estimate the model's performance on the testing set.

Each step is controlled through the [config file](config/config.yaml), which is validated leveraging [pydantic](https://docs.pydantic.dev/latest/).

The notebook [modelling.ipynb](notebooks/modelling.ipynb) provides the framework to train an instance of the FraudDetectionModel. You just have to dump the training and testing data files inside the _data_ folder.

Once the files are dumped, you can play around with the config and different modelling configurations. Each modelling iteration should be tracked locally using [MLFlow](https://mlflow.org/). If you want to look at the best model at the end of your experiments, you can run:

```bash
mlflow ui --port 8080 --backend-store-uri sqlite:///data/mlruns.db
```

This should give you an overview of the best models, and the corresponding configuration.

However, since there is already the necessary modelling files to run the inference (corresponding to the FeatureEngineer and the FraudDetecionModel objects located inside the _models_ folder), you can skip this step to run the inference.

### API

The inference can be run using the corresponding REST API. I leveraged [Flask](https://flask.palletsprojects.com/en/stable/) to build the API, and [Docker](https://www.docker.com/) to contenairize it and serve it locally.

To Dockerize the application, run the following steps:

```bash
docker build -t fraud-detection-api .
```

Then run:

```bash
docker run -p 5001:5001 fraud-detection-api
```

Having followed the steps, the API should be up and running.

If you want to change the model to run the inference, you can change the URI inside the [config file](config/config.py)

You can query directly the endpoint using the terminal, but a user interface is also available to communicate with the API (see next section).

### User interface

I chose to build a small UI leveraging [Streamlit](https://streamlit.io/). After having served the API, you can run the following commands to run the Streamlit interface locally:

```bash
export PYTHONPATH=.
```

And then:

```bash
streamlit run lib/streamlit_app/main.py
```

The app should open and you should be able to input values for the claim you want to inspect. You then just have to click on the "Run Inference" button to get the prediction from the model. You should also be provided with the corresponding SHAP analysis to understand better the prediction.

## Development and Testing

### Testing

The unit tests are built using [pytest](https://docs.pytest.org/en/stable/). Unit tests ensure the robustness of data processing, feature engineering, and model predictions. They are located inside the _tests_ foler. To run tests:

```bash
pytest tests/
```

### Linting and Formatting

This project uses Ruff for linting, Black for formatting, and isort for sorting imports. All checks can be run with:

```bash
make lint
```

Pre-commit hooks help maintain consistent code quality, and can be installed through the Makefile, by running:

```bash
make install_precommit
```

## Repository Structure

```
.
├── .github    <- GitHub Actions workflows
├── bin        <- Bash files
├── config     <- Configuration files
├── data       <- Everything data-related (files / MLFlow db)
├── lib        <- Python modules
├── models     <- To store the modelling objects
├── notebooks  <- Jupyter notebooks
├── secrets    <- Secret files (ignored by git)
└── tests      <- Unit tests
```
