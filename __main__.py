from model.validator import validate_model
from model.data import DataHandler as dhlr


def run_original():
    print('Loading data ...')
    X, y = dhlr.load_data(threshold = 0)

    print('Loaded {0} objects'.format(len(y)))
    best_thres = validate_model(X, y)
    print('')

def run_hybrid():
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from model.multiple_outputs import MultipleOutputClassifier

    def pipeline(self, threshold = 0.05):
        est = make_pipeline(
            CountVectorizer(binary=False, analyzer='word', stop_words='english', max_features = 200000, min_df = 0.00001, ngram_range = (1, 3)),
            TfidfTransformer(norm = 'l2', smooth_idf = True, use_idf = True),
            MultipleOutputClassifier(
                SGDClassifier(loss='log', alpha=0.0001, max_iter=50, penalty = 'l2', tol=None), 
                threshold
            )
        )
        return est

    print('Loading data ...')
    X, y = dhlr.load_data(threshold = 500)

    print('Loaded {0} objects'.format(len(y)))
    best_thres = validate_model(X, y, pipeline)
    print()



def main():
    # print('Running original solution')
    # run_original()

    print('Running hybrid solution')
    run_hybrid()

if __name__ == '__main__':
    main()
