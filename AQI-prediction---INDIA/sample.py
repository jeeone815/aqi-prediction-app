import joblib
import pandas as pd
from util.aqi_utils import aqi_category, predict_aqi


# Load models
xgb_model = joblib.load("models/xgboost_aqi_FineTune_model.pkl")
lgb_model = joblib.load("models/lightgbm_aqi_FineTune_model.pkl")

# Load preprocessing artifacts
feature_names = joblib.load("post-process/feature_names.pkl")
label_encoders = joblib.load("post-process/label_encoders.pkl")

# Sample input
df = pd.read_csv("data/processed/processed_aqi_data.csv")
sample_data = df.drop(columns=["AQI","datetime"]).iloc[-1]
sample_data = sample_data[feature_names].to_dict()
sample_data.pop("AQI_target", None)  # Remove AQI_target — not a model feature

print(sample_data)

# Base model predictions
pred_xgb = predict_aqi(sample_data, xgb_model, label_encoders, feature_names)
print("prediction XGB: ", pred_xgb)
pred_lgb = predict_aqi(sample_data, lgb_model, label_encoders, feature_names)
print("prediction lgb: ", pred_lgb)

# Hybrid prediction
final_pred = (0.8 * pred_lgb) + (0.2 * pred_xgb)


# Output
print("Predicted AQI:", round(final_pred, 2))
print("AQI Category:", aqi_category(final_pred))
