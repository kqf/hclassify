import unittest

from sklearn.ensemble import GradientBoostingClassifier
from model.data import DataHandlerSingle
from model.training import TrainingBundle

from sklearn.linear_model import SGDClassifier
import numpy as np

class TestBT(unittest.TestCase):

    def setUp(self):
        alg = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, min_samples_split = 20)

        params = {
                'gradientboostingclassifier__min_samples_split': [20], 
                'countvectorizer__ngram_range': [(1, 3)]
        }
        self.bundle = TrainingBundle('bt', alg, params, nfolds = 3, n_jobs = 40)


    def test_reproduces_initial_results(self):
        X, y = DataHandlerSingle.load_data()

        print('Starting analysis ...')
        print('Data size', len(y[::5]), '\n\n')

        self.bundle.train_model(X[::5], y[::5])

 