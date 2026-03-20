from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# Base configuration with some hyperparameteres to remove memory allocation error
# change the base config as required and possible 

def get_models():

    models = {

        "Linear Regression": LinearRegression(),

        "Ridge Regression": Ridge(),

        "Lasso Regression": Lasso(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=12,
            min_samples_split=20
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=5,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=50,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=5,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5
        ),

        "AdaBoost": AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        ),

        "LightGBM": LGBMRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            n_jobs=-1
        ),

        "KNN": KNeighborsRegressor(n_neighbors=5)
    }

    scaled_models = [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "KNN"
    ]

    return models, scaled_models