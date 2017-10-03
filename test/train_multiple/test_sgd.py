import unittest

from model.data import DataHandlerStrickt
from model.training import MultiLabelTrainingBundle

from collections import Counter
from sklearn.linear_model import SGDClassifier
import numpy as np

class TestSGD(unittest.TestCase):

    def setUp(self):
        alg = SGDClassifier(loss='squared_hinge', alpha = 1e-5, max_iter=325, tol=None, penalty = 'l2')
        params = {
                'onevsrestclassifier__estimator__learning_rate': ['optimal', 'invscaling'],
                'onevsrestclassifier__estimator__max_iter': [325],
                'onevsrestclassifier__n_jobs': [40],
                'countvectorizer__ngram_range': [(1, 3)]
                # 'countvectorizer__max_features': [1000, 10000, 100000, 200000],
                # 'countvectorizer__min_df': [0.000001, 0.00005, 0.00001]
        }
        self.bundle = MultiLabelTrainingBundle('sgd', alg, params, nfolds = 3, n_jobs = 1)


    def test_reproduces_initial_results(self):
        X, y = DataHandlerStrickt.load_data(threshold = 10)

        print('Starting analysis ...')
        print('Data size', len(y), '\n\n')
        print('Unique objects', len(Counter(tuple(map(frozenset, y)))), '\n\n')
        self.bundle.train_model(X, y)

