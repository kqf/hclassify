import unittest

from model.data import DataHandlerStrickt
from model.training import MultiLabelTrainingBundle


from sklearn.ensemble import RandomForestClassifier
import numpy as np

class TestRF(unittest.TestCase):

    def setUp(self):
        alg = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, min_samples_split=40,
                                                  n_jobs=1, random_state = None)

        params = {
                'onevsrestclassifier__estimator__n_estimators': [100, 150], 
                'countvectorizer__ngram_range': [(1, 3)]
        }
        self.bundle = MultiLabelTrainingBundle('rf', alg, params, nfolds = 3, n_jobs = 40)


    def test_reproduces_initial_results(self):
        X, y = DataHandlerStrickt.load_data(threshold = 10)

        print('Starting analysis ...')
        print('Data size', len(y), '\n\n')

        self.bundle.train_model(X, y)

 