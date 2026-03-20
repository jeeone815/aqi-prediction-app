import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from util.metric import load_metrics


def train_models(models,X_train,X_test,y_train,y_test):

    results={}

    for name,model in models.items():

        print(f"Training {name} model")

        start=time.time()

        model.fit(X_train,y_train)

        preds=model.predict(X_test)

        elapsed=time.time()-start

        mae= mean_absolute_error(y_test,preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2= r2_score(y_test,preds)

        metrics = {
          "MAE": round(mae, 4),
          "RMSE": round(rmse, 4),
          "R2": round(r2, 4),
          "Training_Time": round(elapsed, 2)
          }
        
        results[name] = metrics
        
        model_filename = name.lower().replace(" ", "_")
        joblib.dump(metrics, f"metrics/{model_filename}_metrics.pkl")
        joblib.dump(model,f"models/{model_filename}.pkl")
        load_metrics(results)

    return results