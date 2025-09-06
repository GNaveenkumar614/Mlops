import os
import pickle

import mlflow
from flask import Flask, request, jsonify

mlflow.set_tracking_uri("http://0.0.0.0:5000") #set tracking uri to mlflow server , it cane be local or remote to database
#load model from best run
RUN_ID = '38c67ff3d9f142c7b5cc504d64480bc3'
logged_model = f'runs:/{RUN_ID}/models_mlflow'  
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    
    df= df.drop(["Month_name", "Weekdaysort"], axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month_names'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.drop('Date', axis=1)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    features = pd.get_dummies(df, columns=categorical_cols)

    

    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('price prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


#output:
'''(base) @GNaveenkumar614 ➜ /workspaces/Mlops/deployment/web-service-mlflow (main) $ python predict.py
Downloading artifacts:   0%|                                             | 0/1 [04:04<?, ?it/s]
Downloading artifacts: 100%|████████████████████████████████████| 5/5 [00:00<00:00, 332.34it/s]
 * Serving Flask app "price prediction" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on all addresses.
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://10.0.11.101:9696/ (Press CTRL+C to quit)
 * Restarting with watchdog (inotify)
Downloading artifacts:   0%|                                             | 0/1 [04:04<?, ?it/s]
Downloading artifacts: 100%|████████████████████████████████████| 5/5 [00:00<00:00, 205.74it/s]
 * Debugger is active!
 * Debugger PIN: 118-829-068'''