import unittest

from sklearn.ensemble import GradientBoostingClassifier
from model.data import DataHandlerSingle
from model.training import TrainingBundle

from sklearn.naive_bayes import MultinomialNB
import numpy as np

class TestNB(unittest.TestCase):

    def setUp(self):
        alg = MultinomialNB()

        params = {
                'multinomialnb__alpha': np.linspace(1e-4, 1e-1, 4), 
                'countvectorizer__ngram_range': [(1, 3)],
                'tfidftransformer__norm': ['l1', 'l2']
                # 'tfidftransformer__stop_words': ['english']
                # 'countvectorizer__sparse': [False]
        }
        self.bundle = TrainingBundle('NB', alg, params, nfolds = 3, n_jobs = 40)


    def test_reproduces_initial_results(self):
        X, y = DataHandlerSingle.load_data(threshold = 1000)

        print('Starting analysis ...')
        print('Data size', len(y), '\n\n')

        self.bundle.train_model(X, y)

 