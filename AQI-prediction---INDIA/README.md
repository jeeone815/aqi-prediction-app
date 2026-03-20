# AQI-prediction - INDIA

## Overview
This project predicts Air Quality Index (AQI) values for cities in India using Machine Learning models.
It processes pollution data such as PM2.5, PM10, NO₂, CO, SO₂, and O₃ along with meteorological features to estimate AQI levels.

The goal is to build a system that can help analyze pollution trends and predict air quality conditions.

## Features
- Data cleaning and preprocessing
- AQI calculation based on Indian standards
- Multiple ML models for prediction
- Model evaluation and comparison
- Scalable project structure

## Requirements 

### Python

- Python 3.10+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm

## Project Structure

```
AQI-prediction-INDIA/
│
├── data/
│   ├── raw/
│   └── preprocessed/
│
├── metrics/
|
├── src/
│   ├── preprocessing.py
│   ├── aqi_calculator.py
│   ├── model_training.py
│   └── model_config.py
│
├── models/
│
├── sample.py           # For Testing purpose Only
├── requirements.txt
├── README.md
└── main.py
```

## Installation
Clone the repository:

```
git clone https://github.com/ridamranjan/AQI-prediction---INDIA.git
cd AQI-prediction---INDIA
```

Create virtual environment:

python -m venv venv
source venv/bin/activate

Install dependencies:

```
pip install -r requirements.txt
```

## Usage
Run the training pipeline:

```
python train_pipeline.py
```

## Dataset

Dataset Link: https://www.kaggle.com/datasets/bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025

The dataset contains air pollution measurements including:

- PM2.5
- PM10
- NO2
- SO2
- CO
- O3

meteorological data including:
- Air temperature at 2 meters above the ground
- Humidity Percentage
- Wind speed measured at a height of 10 meters above the ground
- Rain

and other data including: 

- city
- state
- latitude
- longitude
- year
- month
- day
- hour
- Festival Period (1: Yes, 0: No)
- Crop Burning Season (1: Yes, 0: No)

Due to size limitations, datasets are not included in this repository.

Place the dataset inside:

data/raw/


## Models Used

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- **XGBoost**
- **LightGBM**
- KNN

Note: SVR is not used due to huge resource requirement

Models are evaluated using:

- MAE
- R² Score
- Accuracy

## Results

| Model    | MAE       | R²         | Accuracy   |
| -------- | --------- | ---------- | ---------- |
| XGBoost  | 23.42     | 0.6165     | 80.05%     |
| LightGBM | 23.37     | 0.6208     | 80.13%     |


## Future Improvements

- Fine Tune best models (Completed)
- EDA (To be done soon)
- Deep Analysis between models (To be done soon)
- Real-time AQI prediction API (To be done soon)
