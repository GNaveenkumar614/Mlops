#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("coffe_price_pred")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)



def read_dataframe():
    path = '/workspaces/Mlops/dataset/Coffe_sales.csv'
    df = pd.read_csv(path)

    df= df.drop(["Month_name", "Weekdaysort"], axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month_names'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.drop('Date', axis=1)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols)

    

    return df


def create_X(df, dv=None):

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv=None):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


def run(df):
    target='money'
    X=df.drop(["money"], axis=1)
    y=df[target]
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 )     

    run_id = train_model(X_train, y_train, X_test, y_test)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    
    args = parser.parse_args()
    df= read_dataframe()

    run_id = run(df)

    with open("run_id.txt", "w") as f:
        f.write(run_id)