import unittest

from model.data import DataHandlerSingle
from model.training import TrainingBundle

from sklearn.linear_model import SGDClassifier
import numpy as np

class TestSGD(unittest.TestCase):

    def setUp(self):
        alg = SGDClassifier(loss='hinge', alpha=1e-3, max_iter=5, tol=None)

        params = {
                'sgdclassifier__alpha': np.linspace(1e-4, 1e-2, 3), 
                'countvectorizer__ngram_range': [(2, 3)]
        }
        self.bundle = TrainingBundle('sgd', alg, params, nfolds = 3, n_jobs = 40)


    def test_reproduces_initial_results(self):
        X, y = DataHandlerSingle.load_data()

        print('Starting analysis ...')
        print('Data size', len(y[::5]), '\n\n')

        self.bundle.train_model(X[::5], y[::5])

 