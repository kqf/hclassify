import unittest

from model.data import DataHandlerSingle
from model.training import MultiLabelTrainingBundle

from sklearn.linear_model import SGDClassifier
import numpy as np

class TestSGD(unittest.TestCase):

    def setUp(self):
        alg = SGDClassifier(loss='squared_hinge', alpha=1e-3, max_iter=50, tol=None)

        params = {
                'sgdclassifier__loss': ['squared_hinge'],
                'sgdclassifier__max_iter': [50],
                'sgdclassifier__alpha': (0.0001),
                'sgdclassifier__penalty': ('l2'), 
                'countvectorizer__ngram_range': [(1, 3)]
        }
        self.bundle = MultiLabelTrainingBundle('sgd', alg, params, nfolds = 2, n_jobs = 40)


    def test_reproduces_initial_results(self):
        X, y = DataHandlerSingle.load_data()

        print('Starting analysis ...')
        print('Data size', len(y), '\n\n')

        self.bundle.train_model(X, y)

 