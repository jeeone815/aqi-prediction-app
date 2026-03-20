import os
import joblib

def load_metrics(results):
  metrics_path = "metrics/all_model_metrics.pkl"

  if os.path.exists(metrics_path):
      all_results = joblib.load(metrics_path)
  else:
      all_results = {}

  all_results.update(results)

  joblib.dump(all_results, metrics_path)