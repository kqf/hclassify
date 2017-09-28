import unittest
import json
from model.data import load_data, DataHandler
from model.validator import validate_model

class TestThresholds(unittest.TestCase):


    def setUp(self):
        with open('config/test.json') as f:
            self.nominal = json.load(f)['thresholds_100']


    def test_reproduces_initial_results(self):
        X, y = load_data()
        # TODO: Fix DataHandler
        # X, y = DataHandler.load_data()
        f1 = validate_model(X[::100], y[::100])

        for f, nominal in zip(f1, self.nominal):
            self.assertEqual(f, nominal)
