import unittest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

class TestIrisModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load dataset
        cls.data = pd.read_csv("data/iris.csv")

        # Basic checks
        cls.assertGreater = unittest.TestCase().assertGreater
        cls.assertEqual = unittest.TestCase().assertEqual
        cls.assertFalse = unittest.TestCase().assertFalse

        # Load trained model
        cls.model = joblib.load("model.h5")

    def test_data_not_empty(self):
        """Check if dataset is loaded and not empty"""
        self.assertFalse(self.data.empty, "Dataset should not be empty")

    def test_data_columns(self):
        """Check if expected columns exist"""
        expected_columns = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'}
        self.assertTrue(expected_columns.issubset(self.data.columns),
                        "Dataset must contain all required columns")

    def test_no_missing_values(self):
        """Check if there are no missing values"""
        self.assertEqual(self.data.isnull().sum().sum(), 0, "Dataset contains missing values")

    def test_model_accuracy_above_threshold(self):
        """Ensure model achieves at least 50% accuracy"""
        X = self.data.drop('species', axis=1)
        y = self.data['species']
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        self.assertGreater(acc, 0.5, f"Model accuracy too low: {acc:.2f}")

if __name__ == "__main__":
    unittest.main()
