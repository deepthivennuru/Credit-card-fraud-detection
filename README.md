# Credit Card Fraud Detection

This project implements a robust credit card fraud detection system using an ensemble of machine learning models.

## Overview

The system uses multiple models including XGBoost, LightGBM, and Neural Networks to detect fraudulent transactions in credit card data. It includes:

- Exploratory Data Analysis (EDA)
- Model Training and Evaluation
- Real-time Prediction API
- Live Monitoring Dashboard

## Dataset

The dataset used in this project is the Credit Card Fraud Detection dataset. Due to its large size, it's not included in this repository. To use this project:

1. Download the dataset from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Place the downloaded `creditcard.csv` file in the `data/` directory of this project

## Project Structure

```
├── data/                    # Add creditcard.csv here
├── notebooks/
│   └── eda.ipynb           # EDA and model training
├── models/                  # Saved model files
├── app/
│   └── main.py             # FastAPI application
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/deepthivennuru/Credit-card-fraud-detection.git
cd Credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. First, complete the EDA and model training in `notebooks/eda.ipynb`
2. Start the API server:
   ```bash
   uvicorn app.main:app --reload --port 8001
   ```
3. Access the API documentation at http://localhost:8001/docs

## Models

The system uses an ensemble of:
- XGBoost
- LightGBM
- Neural Network

Performance metrics and model comparisons are available in the notebook.

## Monitoring

Includes real-time monitoring of:
- Prediction distribution
- Response times
- Fraud ratio
- System health
