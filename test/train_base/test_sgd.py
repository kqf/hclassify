import unittest

from model.data import DataHandlerSingle
from model.training import TrainingBundle

from sklearn.linear_model import SGDClassifier
import numpy as np

class TestSGD(unittest.TestCase):

    def setUp(self):
        alg = SGDClassifier(loss='squared_hinge', alpha=0.0001, max_iter=60, penalty = 'l2', tol=None)

        params = {
                'countvectorizer__ngram_range': [(1, 3)],
                'countvectorizer__max_features': [50000, 200000, 100000, 200000, 300000],
                'countvectorizer__min_df': [0.00001, 0.0005]
        }
        self.bundle = TrainingBundle('sgd', alg, params, nfolds = 3, n_jobs = 40)


    def test_reproduces_initial_results(self):
        X, y = DataHandlerSingle.load_data()

        print('Starting analysis ...')
        print('Data size', len(y), '\n\n')

        self.bundle.train_model(X, y)

 