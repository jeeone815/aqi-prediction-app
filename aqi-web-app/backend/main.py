import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from backend.config import MODELS_DIR, METRICS_DIR, POSTPROCESS_DIR

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(title="AQI Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Load artifacts once at startup
# ─────────────────────────────────────────────
xgb_model      = joblib.load(os.path.join(MODELS_DIR, "xgboost_aqi_FineTune_model.pkl"))
lgb_model      = joblib.load(os.path.join(MODELS_DIR, "lightgbm_aqi_FineTune_model.pkl"))
feature_names  = joblib.load(os.path.join(POSTPROCESS_DIR, "feature_names.pkl"))
label_encoders = joblib.load(os.path.join(POSTPROCESS_DIR, "label_encoders.pkl"))
all_metrics    = joblib.load(os.path.join(METRICS_DIR, "all_model_metrics.pkl"))

# Strip AQI_target from feature_names (was saved accidentally during training)
SAFE_FEATURES = [f for f in feature_names if f != "AQI_target"]

# City → State mapping (for auto-fill convenience)
CITY_STATE_MAP = {
    "Agartala": "Tripura", "Ahmedabad": "Gujarat", "Aizawl": "Mizoram",
    "Bengaluru": "Karnataka", "Bhopal": "Madhya Pradesh", "Bhubaneswar": "Odisha",
    "Chandigarh": "Punjab", "Chennai": "Tamil Nadu", "Dehradun": "Uttarakhand",
    "Delhi": "Delhi", "Gangtok": "Sikkim", "Gurugram": "Haryana",
    "Guwahati": "Assam", "Hyderabad": "Telangana", "Imphal": "Manipur",
    "Itanagar": "Arunachal Pradesh", "Jaipur": "Rajasthan", "Kohima": "Nagaland",
    "Kolkata": "West Bengal", "Lucknow": "Uttar Pradesh", "Mumbai": "Maharashtra",
    "Panaji": "Goa", "Patna": "Bihar", "Raipur": "Chhattisgarh",
    "Ranchi": "Jharkhand", "Shillong": "Meghalaya", "Shimla": "Himachal Pradesh",
    "Thiruvananthapuram": "Kerala", "Visakhapatnam": "Andhra Pradesh",
}

# City/State encodings using the saved label encoders
CITIES  = list(label_encoders["City"].classes_)
STATES  = list(label_encoders["State"].classes_)

# ─────────────────────────────────────────────
# AQI category helper
# ─────────────────────────────────────────────
def aqi_category(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else:            return "Severe"

# ─────────────────────────────────────────────
# Request schema — user fills these fields
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    City: str
    State: str
    Latitude: Optional[float] = None
    Longitude: Optional[float] = None
    Year: int
    Month: int
    Day: int
    Hour: int
    Temp_2m_C: float
    Humidity_Percent: float
    Wind_Speed_10m_kmh: float
    Is_Raining: int = 0
    Pressure_MSL_hPa: float
    PM2_5_ugm3: float
    PM10_ugm3: float
    CO_ugm3: float
    NO2_ugm3: float
    SO2_ugm3: float
    O3_ugm3: float
    Dust_ugm3: float = 1.0

# City → lat/lon lookup table (approximate centroids)
CITY_COORDS = {
    "Agartala": (23.8315, 91.2868), "Ahmedabad": (23.0225, 72.5714),
    "Aizawl": (23.7271, 92.7176), "Bengaluru": (12.9716, 77.5946),
    "Bhopal": (23.2599, 77.4126), "Bhubaneswar": (20.2961, 85.8245),
    "Chandigarh": (30.7333, 76.7794), "Chennai": (13.0827, 80.2707),
    "Dehradun": (30.3165, 78.0322), "Delhi": (28.7041, 77.1025),
    "Gangtok": (27.3389, 88.6065), "Gurugram": (28.4595, 77.0266),
    "Guwahati": (26.1445, 91.7362), "Hyderabad": (17.3850, 78.4867),
    "Imphal": (24.8170, 93.9368), "Itanagar": (27.0844, 93.6053),
    "Jaipur": (26.9124, 75.7873), "Kohima": (25.6751, 94.1086),
    "Kolkata": (22.5726, 88.3639), "Lucknow": (26.8467, 80.9462),
    "Mumbai": (19.0760, 72.8777), "Panaji": (15.4909, 73.8278),
    "Patna": (25.5941, 85.1376), "Raipur": (21.2514, 81.6296),
    "Ranchi": (23.3441, 85.3096), "Shillong": (25.5788, 91.8933),
    "Shimla": (31.1048, 77.1734), "Thiruvananthapuram": (8.5241, 76.9366),
    "Visakhapatnam": (17.6868, 83.2185),
}

# ─────────────────────────────────────────────
# Seasonal / engineered feature helpers
# ─────────────────────────────────────────────
def is_festival_period(month: int, day: int) -> int:
    # Oct-Nov (Diwali window) and Jan (New Year) treated as festival periods
    return 1 if (month in [10, 11] or (month == 1 and day <= 5)) else 0

def is_crop_burning(month: int) -> int:
    return 1 if month in [10, 11] else 0

def is_winter(month: int) -> int:
    return 1 if month in [11, 12, 1, 2] else 0

def is_rush_hour(hour: int) -> int:
    return 1 if hour in [7, 8, 9, 17, 18, 19] else 0

# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/metrics")
def get_metrics():
    """Return all model metrics (MAE, RMSE, R2, Training_Time)."""
    # Add accuracy % = R2 * 100 for display
    result = {}
    for model, m in all_metrics.items():
        result[model] = {
            "MAE": round(m["MAE"], 4),
            "RMSE": round(m["RMSE"], 4),
            "R2": round(m["R2"], 4),
            "Accuracy": round(m["R2"] * 100, 2),
            "Training_Time_s": round(m["Training_Time"], 2),
        }
    return result

@app.get("/cities")
def get_cities():
    return {"cities": CITIES, "city_state_map": CITY_STATE_MAP}

@app.post("/predict")
def predict(req: PredictRequest):
    """Run hybrid XGBoost + LightGBM AQI prediction."""
    # Resolve lat/lon if not provided
    lat, lon = CITY_COORDS.get(req.City, (20.5937, 78.9629))
    if req.Latitude is not None: lat = req.Latitude
    if req.Longitude is not None: lon = req.Longitude

    # Apply seasonal/engineered features
    festival  = is_festival_period(req.Month, req.Day)
    crop_burn = is_crop_burning(req.Month)
    winter    = is_winter(req.Month)
    rush_hour = is_rush_hour(req.Hour)

    # Encode City and State using label encoders
    city_enc  = label_encoders["City"].transform([req.City])[0]
    state_enc = label_encoders["State"].transform([req.State])[0]

    # Build base row — lag/rolling features approximated from current values
    p25  = req.PM2_5_ugm3
    p10  = req.PM10_ugm3
    o3   = req.O3_ugm3
    # Use current value as a reasonable approximation for lag/rolling features
    row = {
        "City": city_enc,
        "State": state_enc,
        "Latitude": lat,
        "Longitude": lon,
        "Year": req.Year,
        "Month": req.Month,
        "Day": req.Day,
        "Hour": req.Hour,
        "Temp_2m_C": req.Temp_2m_C,
        "Humidity_Percent": req.Humidity_Percent,
        "Wind_Speed_10m_kmh": req.Wind_Speed_10m_kmh,
        "Is_Raining": req.Is_Raining,
        "Pressure_MSL_hPa": req.Pressure_MSL_hPa,
        "PM2_5_ugm3": p25,
        "PM10_ugm3": p10,
        "CO_ugm3": req.CO_ugm3,
        "NO2_ugm3": req.NO2_ugm3,
        "SO2_ugm3": req.SO2_ugm3,
        "O3_ugm3": o3,
        "Dust_ugm3": req.Dust_ugm3,
        "Festival_Period": festival,
        "Crop_Burning_Season": crop_burn,
        "Winter": winter,
        "Rush_Hour": rush_hour,
        # Lag/rolling features — approximated from current values
        "PM2_5_ugm3_lag1": p25,
        "PM2_5_ugm3_lag3": p25,
        "PM2_5_ugm3_roll6": p25,
        "PM2_5_ugm3_roll12": p25,
        "PM10_ugm3_lag1": p10,
        "PM10_ugm3_lag3": p10,
        "PM10_ugm3_roll6": p10,
        "PM10_ugm3_roll12": p10,
        "O3_ugm3_lag1": o3,
        "O3_ugm3_lag3": o3,
        "O3_ugm3_roll6": o3,
        "O3_ugm3_roll12": o3,
        "AQI_lag1": 0.0,  # No prior AQI available for fresh input
        "AQI_lag2": 0.0,
    }

    df = pd.DataFrame([row])[SAFE_FEATURES]

    pred_xgb = float(xgb_model.predict(df)[0])
    pred_lgb = float(lgb_model.predict(df)[0])
    final    = round((0.8 * pred_lgb) + (0.2 * pred_xgb), 2)

    return {
        "predicted_aqi": final,
        "category": aqi_category(final),
        "xgb_prediction": round(pred_xgb, 2),
        "lgb_prediction": round(pred_lgb, 2),
    }

# ─────────────────────────────────────────────
# Serve frontend static files
# ─────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
