import unittest

from model.data import DataHandlerSingle
from model.training import MultiLabelTrainingBundle

from sklearn.linear_model import SGDClassifier
import numpy as np

class TestSGD(unittest.TestCase):

    def setUp(self):
        alg = SGDClassifier(loss='squared_hinge', alpha = 0.0001, max_iter=50, tol=None, penalty = 'l2')
        params = {
                'onevsrestclassifier__estimator__max_iter': [40, 50, 60, 70, 80, 100],
                'onevsrestclassifier__n_jobs': [40],
                'countvectorizer__ngram_range': [(1, 3)],
                'countvectorizer__max_features': [100000, 200000, 200000],
                'countvectorizer__min_df': [0.00001, 0.00005, 0.00001]
        }
        self.bundle = MultiLabelTrainingBundle('sgd', alg, params, nfolds = 2, n_jobs = 1)


    def test_reproduces_initial_results(self):
        X, y = DataHandlerSingle.load_data()

        print('Starting analysis ...')
        print('Data size', len(y), '\n\n')
        self.bundle.train_model(X, y)

