
# # from fastapi import FastAPI
# # from pydantic import BaseModel

# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import mean_squared_error
# # import pandas as pd
# # import joblib

# # from model import IrisModel, IrisSpecies


# # #Créer une instance de FastAPI
# # app = FastAPI()
# # model = joblib.load("iris_regressor.pkl")

# # # Définir un modèle de requête
# # class IrisRequest(BaseModel):
# #     sepal_width: float
# #     petal_length: float
# #     petal_width: float

# # # Charger le modèle lors du démarrage de l'application
# # #@app.on_event("startup")
# # #def load_model():
# # #    global model


# # # Point de terminaison pour faire une prédiction
# # @app.post("/predictpost")
# # def predictpost(iris: IrisRequest):
# #     # Convertir la requête en un tableau NumPy
# #     data = [[iris.sepal_width, iris.petal_length, iris.petal_width]]

# #     # Faire une prédiction
# #     prediction = model.predict(data)

# #     # Retourner la prédiction
# #     return {"predicted_sepal_length": prediction[0]}

# # # Point de terminaison pour vérifier que l'API est en cours d'exécution
# # @app.get("/status")
# # def status():
# #     return {"message": "API is up and running!"}

# # @app.get("/predictget")
# # def predictget(sepal_width: float, petal_length: float, petal_width: float):
# #     # Convertir les paramètres en un tableau NumPy
# #     data = [[sepal_width, petal_length, petal_width]]

# #     # Faire une prédiction
# #     prediction = model.predict(data)

# #     # Retourner la prédiction
# #     return {"predicted_sepal_length": prediction[0]}


# # # @app.get('/predictpost')
# # # def predict_species(iris: IrisRequest):
# # #     print('im here')
# # #     data = iris.dict()
# # #     prediction, probability = model.predict_species(
# # #         data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']
# # #     )
# # #     return {
# # #         'prediction': prediction,
# # #         'probability': probability
# # #     }

# from fastapi import FastAPI
# from pydantic import BaseModel
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import joblib



# #Créer une instance de FastAPI
# app = FastAPI()
# model = joblib.load("iris_model.pkl")

# # Définir un modèle de requête
# class IrisRequest(BaseModel):
#     sepal_width: float
#     petal_length: float
#     petal_width: float

# # # Charger le modèle lors du démarrage de l'application
# # @app.on_event("startup")
# # def load_model():
# #     global model


# # Point de terminaison pour faire une prédiction
# @app.post("/predict")
# def predict(iris: IrisRequest):
#     # Convertir la requête en un tableau NumPy
#     data = [[iris.sepal_width, iris.petal_length, iris.petal_width]]

#     # Faire une prédiction
#     prediction = model.predict(data)

#     # Retourner la prédiction
#     return {"predicted_sepal_length": prediction[0]}

# api.py
from fastapi import FastAPI
import numpy as np
from model import IrisModel  # Importer la classe IrisModel

# Charger le modèle en utilisant la classe IrisModel
iris_model = IrisModel()
iris_model.load_model()

# Initialiser FastAPI
app = FastAPI()

# Route pour faire une prédiction avec une requête GET
@app.post("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    # Convertir les paramètres en un tableau NumPy
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Faire une prédiction en utilisant la classe IrisModel
    prediction = iris_model.predict(data)

    # Retourner la classe prédite
    return {"predicted_class": int(prediction[0])}
