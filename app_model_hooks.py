import os
import pickle
import subprocess

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config["DEBUG"] = True


# Enruta la landing page (endpoint /)
@app.route("/", methods=["GET"])
def hello():
    return "Bienvenido a mi API del modelo endgagementpredictor:\n Puedes acceder a nuestro modelo poniendo los siguientes endpoints:\n - Para reentrenar: \n - Para predecir:"


# Enruta la funcion al endpoint /api/v1/predict

# Variables: SessionsPerWeek, AvgSessionDurationMinutes, AchievementsUnlocked

"""
La petición de prueba sería:
??? REVISAR LA URL --> http://127.0.0.1:5000/api/v1/predict?SessionsPerWeek=15&AvgSessionDurationMinutes=60&AchievementsUnlocked=80
"""


@app.route("/api/v1/predict", methods=["GET"])
def predict():  # Ligado al endpoint '/api/v1/predict', con el método GET
    model = pickle.load(open("model.pkl", "rb"))
    SessionsPerWeek = request.args.get("SessionsPerWeek", None)
    AvgSessionDurationMinutes = request.args.get("AvgSessionDurationMinutes", None)
    AchievementsUnlocked = request.args.get("AchievementsUnlocked", None)

    print(SessionsPerWeek, AvgSessionDurationMinutes, AchievementsUnlocked)
    # print(type(tv))

    if SessionsPerWeek is None or AvgSessionDurationMinutes is None or AchievementsUnlocked is None:
        return "Args empty, the data are not enough to predict, STUPID!!!!"
    else:
        api_get = {'SessionsPerWeek': float(SessionsPerWeek),
           'AchievementsUnlocked': float(AvgSessionDurationMinutes),
           'AvgSessionDurationMinutes': float(AchievementsUnlocked)}
        df_get = pd.DataFrame(api_get, index=[0])
        prediction = model.predict(df_get)

    return jsonify({"predictions": prediction[0]})
   


"""
La petición de prueba sería:
??? REVISAR LA URL --> http://127.0.0.1:5000/api/v1/retrain
"""


@app.route("/api/v1/retrain", methods=["GET"])
# Enruta la funcion al endpoint /api/v1/retrain
def retrain():  # Rutarlo al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists("data/online_gaming_behavior_dataset_nuevos.csv"):
        data = pd.read_csv("data/online_gaming_behavior_dataset_nuevos.csv")

        num_col = ['SessionsPerWeek', 'AvgSessionDurationMinutes', 'AchievementsUnlocked']
        target = 'EngagementLevel'

        train_set, test_set = train_test_split(data, test_size=0.2, stratify=data[target], random_state=42)

        X_train = train_set[num_col]
        X_test = test_set[num_col]

        y_train = train_set[target]
        y_test = test_set[target]

        labels = {'Low':0,
                'Medium': 1,
                'High': 2}
        y_train_encoded = y_train.map(labels)
        y_test_encoded = y_test.map(labels)

        #Preprocessing
        num_pipe = Pipeline([
            ('mmscaler', MinMaxScaler())
        ])

        preprocessing = ColumnTransformer([
            ('normalize', num_pipe, num_col)],
            remainder='passthrough').set_output(transform= 'pandas')


        #Entrenamiento
        logreg_pipe = Pipeline([
            ('preprocessing', preprocessing),
            ('modelo', LogisticRegression(class_weight='balanced', C=1.0, max_iter=50, 
                                        solver='saga', random_state=42))])

        logreg_pipe.fit(X_train, y_train_encoded)
        
        pickle.dump(logreg_pipe, open("ad_model.pkl", "wb"))

        return f"Model retrained. New evaluation with BALANCED ACCURACY: {logreg_pipe.score()}" ## !! Revisar

    else:
        return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>"


@app.route("/webhook", methods=["POST"])
def webhook():
    # Ruta al repositorio donde se realizará el pull
    path_repo = "/home/tc24/tcflask"
    servidor_web = "/var/www/tc24_pythonanywhere_com_wsgi.py"

    # Comprueba si la solicitud POST contiene datos JSON
    if request.is_json:
        payload = request.json
        # Verifica si la carga útil (payload) contiene información sobre el repositorio
        if "repository" in payload:
            # Extrae el nombre del repositorio y la URL de clonación
            repo_name = payload["repository"]["name"]
            clone_url = payload["repository"]["clone_url"]

            # Cambia al directorio del repositorio
            try:
                os.chdir(path_repo)
            except FileNotFoundError:
                return jsonify(
                    {"message": "El directorio del repositorio no existe!"}
                ), 404

            # Realiza un git pull en el repositorio
            try:
                subprocess.run(["git", "pull", clone_url], check=True)
                subprocess.run(
                    ["touch", servidor_web], check=True
                )  # Trick to automatically reload PythonAnywhere WebServer
                return jsonify(
                    {"message": f"Se realizó un git pull en el repositorio {repo_name}"}
                ), 200
            except subprocess.CalledProcessError:
                return jsonify(
                    {
                        "message": f"Error al realizar git pull en el repositorio {repo_name}"
                    }
                ), 500
        else:
            return jsonify(
                {
                    "message": "No se encontró información sobre el repositorio en la carga útil (payload)"
                }
            ), 400
    else:
        return jsonify({"message": "La solicitud no contiene datos JSON"}), 400


if __name__ == "__main__":
    app.run()
