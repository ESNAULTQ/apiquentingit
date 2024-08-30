# import unittest


# from model import IrisModel, IrisSpecies


# class TestIrisModel(unittest.TestCase):
#    def test_model_initialization(self):
#        new_model = IrisModel()
#        self.assertIn('iris_model.joblib',new_model.model_fname_)

import unittest
from model import IrisModel

class TestIrisModel(unittest.TestCase):
    def setUp(self):
        # Initialiser une instance d'IrisModel avant chaque test
        self.model = IrisModel()

    def test_model_initialization(self):
        # Vérifier que le nom du fichier modèle est correct
        self.assertIn('iris_model.pkl', self.model.model_fname)

    def test_model_loading(self):
        # Charger le modèle et vérifier qu'il n'est pas None
        self.model.load_model()
        self.assertIsNotNone(self.model.model)

    def test_model_prediction(self):
        # Charger le modèle avant de faire des prédictions
        self.model.load_model()

        # Exemple de données pour prédiction
        sample_data = [[5.1, 3.5, 1.4, 0.2]]  # Valeurs arbitraires correspondant aux dimensions des fleurs

        # Faire une prédiction et vérifier que le résultat est un entier (classe prédite)
        prediction = self.model.predict(sample_data)
        self.assertIsInstance(prediction[0], int)

if __name__ == '__main__':
    unittest.main()
