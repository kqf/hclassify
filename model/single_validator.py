from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
import numpy as np


def model_definition_words():
    est = make_pipeline(
        CountVectorizer(binary=False, analyzer='word'),
        TfidfTransformer(),
        LinearSVC(),
        # RandomForestClassifier(n_estimators=100, min_samples_leaf=20, min_samples_split=40,
                                                  # n_jobs=1, random_state = 1)
    )
    return est

def validate_model(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=1)
    est = model_definition_words()
    print (est.named_steps.keys())
    est.fit(X_tr, y_tr)

    train_predictions = est.predict(X_tr)
    test_predictions = est.predict(X_te)

    print('On test sample accuracy:', metrics.accuracy_score(y_te, test_predictions))
    print(metrics.classification_report(y_te, test_predictions, y_te))
    print()
    print()
    print('On training sample accuracy:', metrics.accuracy_score(y_tr, train_predictions))
    print(metrics.classification_report(y_tr, train_predictions, y_tr))

    print (y_te[0], test_predictions[0])

def train_model(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    cv_strategy = StratifiedShuffleSplit(n_splits = 5, test_size = 0.3)
    # cv_strategy = KFold(3)

    params = {
                'linearsvc__C': np.linspace(1e-3, 1e2, 5), 
                'countvectorizer__ngram_range': [(1, 1), (1, 2)]
             }

    estimator = model_definition_words()
    search = GridSearchCV(estimator, params, cv = cv_strategy, verbose = 10, n_jobs = 40)
    search.fit(X_tr, y_tr)

    est = search.best_estimator_
    train_predictions = est.predict(X_tr)
    test_predictions = est.predict(X_te)

    print('On test sample accuracy:', metrics.accuracy_score(y_te, test_predictions))
    print(metrics.classification_report(y_te, test_predictions, y_te))
    print()
    print()
    print('On training sample accuracy:', metrics.accuracy_score(y_tr, train_predictions))
    # print(metrics.classification_report(y_tr, train_predictions, y_tr))
