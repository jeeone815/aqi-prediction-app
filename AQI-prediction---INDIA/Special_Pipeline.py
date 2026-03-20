import pandas as pd
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import r2_score, mean_absolute_error 
from HyperParameters.FineTuning import new_data, xgb_hper, lgbm_hyper

# Turn True before running if there is a change in preprocessing (Only once)

new_data_bool = False

# Turn True before running if there is a increase in Accuracy (Only once)

run_tuning_bool = False

# Put verbose = 0, 1, 2, 3 based on how detailed you want hypertuning
# Reduce CV or n_iter if it's taking too much time
# Change n_jobs only increase if your proccesor can handle it 

n_jobs = 2 
cv = 3
verbose = 0
n_iter = 30
step = True # change for Basic step log

#############################################
## DO NOT TOUCH THE BELOW CODE###############


new_data(new_data_bool)
df = pd.read_csv("data/processed/processed_aqi_data.csv")

if step:
    print("Data Loading Done")

X = df.drop(columns=["AQI", "datetime", "AQI_target"])
Y = df["AQI_target"]

x_train, x_test, y_train, y_test = train_test_split(
  X, Y, test_size=0.2, shuffle=False
)

if step:
    print("Data Splitting Done")
    print("\n\n")

mean_aqi = y_test.mean()

# The Below parameters are found after carfeul testing and many conisderation

xg_para = xgb_hper(run_tuning_bool, X, Y, n_jobs, cv, verbose, n_iter, step)

xgb_model = XGBRegressor(
    
    **xg_para,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

xgb_model.fit(x_train, y_train)
pred_xgb = xgb_model.predict(x_test)

mae_xgb = mean_absolute_error(y_test, pred_xgb)
r2_xgb = r2_score(y_test, pred_xgb)

acc_xgb = (1 - (mae_xgb / mean_aqi)) * 100

print("XGBoost MAE:", mae_xgb)
print("XGBoost R2:", r2_xgb)
print("XGBoost Accuracy:", acc_xgb, "%")
print("\n\n")

joblib.dump(xgb_model, "models/xgboost_aqi_FineTune_model.pkl")

# The Below parameters are found after carfeul testing and many conisderation

lg_para = lgbm_hyper(run_tuning_bool, X, Y, n_jobs, cv, verbose, n_iter, step)

lgb_model = lgb.LGBMRegressor(

    **lg_para,
    n_jobs=-1,
    verbosity=-1,
    random_state=42
)

lgb_model.fit(x_train, y_train)
pred_lgb = lgb_model.predict(x_test)

joblib.dump(lgb_model, "models/lightgbm_aqi_FineTune_model.pkl")

mae_lgb = mean_absolute_error(y_test, pred_lgb)
r2_lgb = r2_score(y_test, pred_lgb)

acc_lgb = (1 - (mae_lgb / mean_aqi)) * 100

print("LightGBM MAE:", mae_lgb)
print("LightGBM R2:", r2_lgb)
print("LightGBM Accuracy:", acc_lgb, "%")