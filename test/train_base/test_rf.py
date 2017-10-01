import unittest

from sklearn.ensemble import RandomForestClassifier
from model.data import DataHandlerSingle
from model.training import TrainingBundle

from sklearn.linear_model import SGDClassifier
import numpy as np

class TestRF(unittest.TestCase):

    def setUp(self):
        alg = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, min_samples_split=40,
                                                  n_jobs=1, random_state = 1)

        params = {
                'randomforestclassifier__n_estimators': [50, 100, 200], 
                'countvectorizer__ngram_range': [(2, 3)]
        }
        self.bundle = TrainingBundle('rf', alg, params, nfolds = 3, n_jobs = 40)


    def test_reproduces_initial_results(self):
        X, y = DataHandlerSingle.load_data()

        print('Starting analysis ...')
        print('Data size', len(y[::5]), '\n\n')

        self.bundle.train_model(X[::5], y[::5])

 