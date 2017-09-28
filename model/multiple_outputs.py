import pandas as pd

from operator import itemgetter
from funcy.seqs import chunks
from sklearn.base import BaseEstimator, ClassifierMixin


class MultipleOutputClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, threshold=0.3):
        self.base_estimator = base_estimator
        self.threshold = threshold

    def fit(self, X, y=None, sample_weight=None):
        self.base_estimator.fit(X, y, sample_weight)
        return self

    def predict(self, X):
        return self.predict_many(X)

    def predict_many(self, X):
        # predicts all the classes with probability >= threshold
        # and the most probable category
        return [self.extract_classes(probs_dict) for probs_dict in self.predict_proba_dict(X)]

    def extract_classes(self, probs_dict):
        # accepts a dictionary and returns a list of strings with classes that exceed the threshold + first class
        probs_sorted = sorted(probs_dict.items(), key=itemgetter(1), reverse=True)
        return [cl for i, (cl, probability) in enumerate(probs_sorted)
                if i == 0 or probability >= self.threshold]

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def predict_proba_dict(self, X):
        # predicts a list of dictionaries
        # iterate over chunks
        for ch in chunks(1000, range(X.shape[0])):
            X_ch = X[ch, :]
            probs_df = pd.DataFrame(self.base_estimator.predict_proba(X_ch), columns=self.base_estimator.classes_)
            for probs in probs_df.T.to_dict().values():
                yield probs

