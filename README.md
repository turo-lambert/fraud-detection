<div align="center">

# fraud-detection-artefact

[![CI status](https://github.com/artefactory-nl/fraud-detection/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory-nl/fraud-detection/actions/workflows/ci.yaml?query=branch%3Amain)
[![Python Version](https://img.shields.io/badge/python-3.10-blue)]()

[![Linting , formatting, imports sorting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory-nl/fraud-detection/blob/main/.pre-commit-config.yaml)
</div>

TODO: if not done already, check out the [Skaff documentation](https://artefact.roadie.so/catalog/default/component/repo-builder-ds/docs/) for more information about the generated repository.

"Boilerplate for Artefact's Data Science team."

## Table of Contents

- [fraud-detection](#fraud-detection)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Documentation](#documentation)
  - [Repository Structure](#repository-structure)

## Installation

To install the required packages in a virtual environment, run the following command:

```bash
make install
```

TODO: Choose between conda and venv if necessary or let the Makefile as is and copy/paste the [MORE INFO installation section](MORE_INFO.md#eased-installation) to explain how to choose between conda and venv.

A complete list of available commands can be found using the following command:

```bash
make help
```

## Usage

TODO: Add usage instructions here

## Repository Structure

```
.
├── .github    <- GitHub Actions workflows and PR template
├── bin        <- Bash files
├── config     <- Configuration files
├── docs       <- Documentation files
├── lib        <- Python modules
├── notebooks  <- Jupyter notebooks
├── secrets    <- Secret files (ignored by git)
└── tests      <- Unit tests
```
