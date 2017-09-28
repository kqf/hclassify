import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline

from model.data import flatten_data
from model.multiple_outputs import MultipleOutputClassifier


def model_definition_words(threshold=0.05):
    est = make_pipeline(
        CountVectorizer(min_df=100, binary=True, analyzer='word'),
        MultipleOutputClassifier(
            base_estimator=RandomForestClassifier(n_estimators=100, min_samples_leaf=20, min_samples_split=40,
                                                  n_jobs=1, random_state = 1),
            threshold=threshold)
    )
    return est

    
def validate_model(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=1)
    X_tr_flat, y_tr_flat = flatten_data(X_tr, y_tr)

    est = model_definition_words(0.05)
    est.fit(X_tr_flat, y_tr_flat)

    best_thres = 0
    best_mean_f1 = 0

    all_f1 = []
    for thres in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
        est.steps[-1][1].threshold = thres
        preds = est.predict(X_te)
        mean_f1 = np.array([f1(true, pred) for true, pred in zip(y_te, preds)]).mean()
        all_f1.append(mean_f1)
        if mean_f1 > best_mean_f1:
            best_thres = thres
            best_mean_f1 = mean_f1
        print('Thres {} avg f1 {}'.format(thres, mean_f1))
    print('Best threshold found {}'.format(best_thres))
    print('Best F1 {}'.format(best_mean_f1))
    # return best_thres
    return all_f1


def f1(true, pred):
    if len(pred) == 0:
        return 0
    tp = len(set(true).intersection(set(pred)))
    precision = tp / len(pred)
    recall = tp / len(true)
    if (precision + recall) == 0:
        return 0
    else:
        return (2 * precision * recall / (precision + recall))


