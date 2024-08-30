
# # 1. Library imports
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from pydantic import BaseModel
# from sklearn.datasets import load_iris
# import joblib



# # 2. Class which describes a single flower measurements
# class IrisSpecies(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float


# # 3. Class for training the model and making predictions
# class IrisModel:
#     # 6. Class constructor, loads the dataset and loads the model
#     #    if exists. If not, calls the _train_model method and
#     #    saves the model
#     def __init__(self):
#         self.data = load_iris(as_frame=True)
#         self.model_fname_ = 'iris_model.pkl'
#         try:
#             self.model = joblib.load(self.model_fname_)
#         except Exception as _:
#             self.model = self._train_model()
#             joblib.dump(self.model, self.model_fname_)


#     # 4. Perform model training using the RandomForest classifier
#     def _train_model(self):
#         X = self.data.data
#         y = self.data.target
#         rfc = RandomForestClassifier()
#         model = rfc.fit(X, y)
#         return model


#     # 5. Make a prediction based on the user-entered data
#     #    Returns the predicted species with its respective probability
#     def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
#         data_in = [[sepal_length, sepal_width, petal_length, petal_width]]
#         prediction = self.model.predict(data_in)
#         probability = self.model.predict_proba(data_in).max()
#         return prediction[0], probability

# # def test_model_initialization(self):
# #     new_model = IrisModel()
# #     self.assertIn('iris_model.pkl', new_model.model_fname_)

# model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class IrisModel:
    def __init__(self):
        self.model_fname = 'iris_model.pkl'
        self.model = None

    def train(self):
        # Charger le dataset Iris
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialiser et entraîner le modèle
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Faire des prédictions et vérifier l'exactitude
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        # Sauvegarder le modèle
        self.save_model()

    def save_model(self):
        joblib.dump(self.model, self.model_fname)
        print(f"Modèle enregistré en tant que '{self.model_fname}'")

    def load_model(self):
        self.model = joblib.load(self.model_fname)
        print(f"Modèle chargé depuis '{self.model_fname}'")

    def predict(self, data):
        return self.model.predict(data)

if __name__ == "__main__":
    iris_model = IrisModel()
    iris_model.train()
