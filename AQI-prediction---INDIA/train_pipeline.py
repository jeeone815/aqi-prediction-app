import pandas as pd
import joblib
import os

from src.preprocessing import process_data
from src.model_training import train_models
from src.model_config import get_models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# LOAD DATA (Original raw data only)
# ============================================================

df = pd.read_csv("data/raw/INDIA_AQI_COMPLETE_20251126.csv")

# ============================================================
# Process DATASET /preprocess.py
# ============================================================


df, label_encoders = process_data(df)

print(df[["PM2_5_ugm3","PM10_ugm3","NO2_ugm3","SO2_ugm3","O3_ugm3","CO_ugm3"]].describe())

# ============================================================
# FEATURE / TARGET SPLIT
# ============================================================

X = df.drop(columns=["AQI", "datetime"])
y = df["AQI"]

feature_names = X.columns.tolist()

# ============================================================
# TRAIN TEST SPLIT (make sure shuffle is always False)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_raw = X_train
X_test_raw = X_test

# ============================================================
# SAVE PREPROCESSING ARTIFACTS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

joblib.dump(scaler, "post-process/scaler.pkl")
joblib.dump(feature_names, "post-process/feature_names.pkl")
joblib.dump(label_encoders, "post-process/label_encoders.pkl")

df.to_csv("data/processed/processed_aqi_data.csv", index=False)

print("Preprocessing complete. Artifacts saved.")

# ============================================================
# DEFINE MODELS /model_config.py
# ============================================================

models, scaled_model_names = get_models()

scaled_models = {k:v for k,v in models.items() if k in scaled_model_names}
tree_models = {k:v for k,v in models.items() if k not in scaled_model_names}

# ============================================================
# Training /model_training.py
# ============================================================

scaled_results = train_models(
    scaled_models,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

tree_results = train_models(
    tree_models,
    X_train_raw,
    X_test_raw,
    y_train,
    y_test
)

# ============================================================
# Result
# ============================================================

results = {**scaled_results, **tree_results}

print("\nModel Results:")
for model, metrics in results.items(): 
    print(model, metrics)