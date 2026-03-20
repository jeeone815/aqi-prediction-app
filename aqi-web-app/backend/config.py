import os

# Absolute path to the AQI-prediction---INDIA project folder
# If you move the project, update this path.
MODEL_BASE = os.environ.get(
    "MODEL_BASE",
    "/home/samantdev/Desktop/finalPro/AQI-prediction---INDIA"
)

MODELS_DIR      = os.path.join(MODEL_BASE, "models")
METRICS_DIR     = os.path.join(MODEL_BASE, "metrics")
POSTPROCESS_DIR = os.path.join(MODEL_BASE, "post-process")
