from model.validator import validate_model, validate_model_multiclass
from model.data import DataHandler as dhlr
from model.data import DataHandlerStrickt as dhlrs

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from model.multiple_outputs import MultipleOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def run_original():
    print('Loading data ...')
    X, y = dhlr.load_data()

    print('Loaded {0} objects'.format(len(y)))
    best_thres = validate_model(X, y)
    print('')

def run_hybrid():
    def pipeline(self, threshold = 0.05):
        est = make_pipeline(
            CountVectorizer(binary=False, analyzer='word', stop_words='english', max_features = 200000, min_df = 0.00001, ngram_range = (1, 3)),
            TfidfTransformer(norm = 'l2', smooth_idf = True, use_idf = True),
            MultipleOutputClassifier(
                SGDClassifier(loss='squared_hinge', alpha = 1e-5, max_iter=325, tol=None, penalty = 'l2'), 
                threshold
            )
        )
        return est

    print('Loading data ...')
    X, y = dhlrs.load_data(threshold = 10)

    print('Loaded {0} objects'.format(len(y)))
    best_thres = validate_model(X, y, pipeline)
    print()


def run_ovr():
    est = make_pipeline(
        CountVectorizer(binary=False, analyzer='word', stop_words='english', ngram_range = (1, 3)),
        TfidfTransformer(norm = 'l2', smooth_idf = True, use_idf = True),
        OneVsRestClassifier(
            SGDClassifier(loss='squared_hinge', alpha = 1e-5, max_iter=325, tol=None, penalty = 'l2') 
            )
    )

    print('Loading data ...')

    mlb = MultiLabelBinarizer()
    X, y = dhlrs.load_data(threshold = 10)
    y = mlb.fit_transform(y)
    validate_model_multiclass(est, X, y, mlb.inverse_transform)




def main():
    print('Running original approach on clean dataset')
    # run_original()

    print('Running hybrid solution')
    # run_hybrid()

    print('Running OvR solution')
    run_ovr()

if __name__ == '__main__':
    main()
