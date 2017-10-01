import unittest

from model.data import DataHandlerSingle
from model.training import TrainingBundle

from sklearn.svm import LinearSVC
import numpy as np

class TestSVM(unittest.TestCase):

    def setUp(self):


        alg = LinearSVC()
        params = {
                'linearsvc__C': np.linspace(1, 1e4, 4), 
                'countvectorizer__ngram_range': [(2, 3)]
        }
        self.bundle = TrainingBundle('svc', alg, params, nfolds = 3, n_jobs = 12)


    def test_reproduces_initial_results(self):
        X, y = DataHandlerSingle.load_data()

        print('Starting analysis ...')
        print('Data size', len(y[::5]), '\n\n')

        self.bundle.train_model(X[::5], y[::5])

 