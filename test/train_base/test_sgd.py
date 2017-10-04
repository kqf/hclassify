import unittest

from model.data import DataHandlerSingle
from model.training import TrainingBundle

from sklearn.linear_model import SGDClassifier
import numpy as np

class TestSGD(unittest.TestCase):

    def setUp(self):
        alg = SGDClassifier(loss='squared_hinge', alpha=0.0001, max_iter=5, penalty = 'l2', tol=None)

        params = {
                # 'sgdclassifier__alpha': np.linspace(1e-6, 1e-2, 10),
                'countvectorizer__binary': [True, False],
                'countvectorizer__ngram_range': [(1, 3)],
                'countvectorizer__max_features': [200000],
                'countvectorizer__min_df': [0.00001,]
        }
        self.bundle = TrainingBundle('sgd', alg, params, nfolds = 3, n_jobs = 40)


    def test_reproduces_initial_results(self):
        X, y = DataHandlerSingle.load_data(threshold = 10)

        print('Starting analysis ...')
        print('Data size', len(y), '\n\n')

        self.bundle.train_model(X, y)

 