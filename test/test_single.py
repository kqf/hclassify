import unittest
import json
from model.data import DataHandlerSingle
from model.single_validator import validate_model, train_model

class TestSingle(unittest.TestCase):

    @unittest.skip('test')
    def test_reproduces_initial_results(self):
        X, y = DataHandlerSingle.load_data()

        print('Starting analysis ...')
        f1 = validate_model(X[::100], y[::100])

        print(f1)

    def test_reproduces_initial_results(self):
        X, y = DataHandlerSingle.load_data()

        print('Starting analysis ...')
        f1 = train_model(X, y)
 