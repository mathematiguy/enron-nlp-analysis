# enron-nlp-analysis
Comp-550 Fall 2023 term group project

# Project Setup and Execution Guide

## Introduction
This README provides instructions for setting up and running a project that uses Data Version Control (DVC). DVC is an open-source tool for data science and machine learning projects. It allows for tracking and versioning of datasets and machine learning models, making it easier to share and reproduce experiments and analyses.

## Prerequisites
Before you start, ensure you have the following installed:
- Python (version as per project requirements)
- pip (Python package manager)
- virtualenv (Python environment management tool)
- DVC (Data Version Control)

## Setup Instructions

### 1. Clone the Project Repository
Clone the project from the provided source and navigate to the project directory.

### 2. Create and Activate a Python Virtual Environment
```
python -m venv venv
```

On Windows: `venv\Scripts\activate`
macOS and Linux: `source venv/bin/activate`

### 3. Install Project Dependencies

```
pip install -r requirements.txt
```

### 4. Download NLTK Dependencies
Run the following in the terminal:

```bash
python -m nltk.downloader -d data/nltk_data all
```

### 5. Install spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

## Running the Project

### Data Version Control (DVC)

Run dvc repro to execute the DVC pipeline. DVC manages the data processing stages as per the dvc.yaml file.

To run the project, run:

```
dvc repro
```

This will run the entire pipeline from start to finish. If you wanna see the dag for the project, run `dvc dag`.