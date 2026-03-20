import pandas as pd
from xgboost import XGBRegressor
import lightgbm as lgb
from src.preprocessing import process_data
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit


# Only change if you know what you are doing

def new_data(bool):

  if bool:

    df = pd.read_csv("data/raw/INDIA_AQI_COMPLETE_20251126.csv")
    df, label_encoders = process_data(df)
    joblib.dump(label_encoders, "post-process/label_encoders.pkl")
    df.to_csv("data/processed/processed_aqi_data.csv", index=False)
    X = df.drop(columns=["AQI", "datetime"])
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "post-process/feature_names.pkl")

  return

def fineTune(model, param, x, y, n_jobs=2, cv=3, verbose=0, n_iter=30, step=False):

  cv = TimeSeriesSplit(n_splits=cv)

  search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param,
    n_iter=n_iter,
    scoring="r2",
    cv=cv,
    verbose=verbose,
    random_state=42,
    n_jobs=n_jobs
  )

  search.fit(x, y)

  if step:
    print(f"{model} is hyperTuned with best parameter {search.best_params_}")

  return search.best_params_

def xgb_hper(bool, x, y, n_jobs, cv, verbose, n_iter, step):

  if bool:

    param_dist_xgb = {
    "n_estimators": [300, 400, 500, 700, 900],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.02, 0.04],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 1.5, 2]
    }
    
    model = XGBRegressor(objective="reg:squarederror", tree_method="hist")
    
    new_parm = fineTune(model, param_dist_xgb, x, y, n_jobs, cv, verbose, n_iter, step)

    joblib.dump(new_parm, "HyperParameters/xgb.pkl")

    return new_parm
  
  else:

    return joblib.load("HyperParameters/xgb.pkl")
  
def lgbm_hyper(bool, x, y, n_jobs, cv, verbose, n_iter, step):

  if bool:

    param_dist_lgb = {
    "n_estimators": [300, 400, 500, 700, 900],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.02, 0.04],
    "num_leaves": [31, 50, 70, 100, 150],
    "max_depth": [-1, 6, 10, 15],
    "min_child_samples": [10, 20, 30, 50],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [0, 0.5, 1]
    }

    model = lgb.LGBMRegressor(objective="regression")

    new_param = fineTune(model, param_dist_lgb, x, y, n_jobs, cv, verbose, n_iter, step)

    joblib.dump(new_param, "HyperParameters/lgb.pkl")

    return new_param

  else:

    return joblib.load("HyperParameters/lgb.pkl")