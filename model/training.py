from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import f1_score, make_scorer
from model.validator import f1_mean

class TrainingBundle():
    ofile_extension = '-single.pkl'
    def __init__(self, name, algorithm, grid_pars, nfolds, n_jobs = 40):
        self.name = name
        self.algorithm = algorithm
        self.grid_pars = grid_pars
        self.nfolds = nfolds
        self.n_jobs = n_jobs
        self.scorer = 'accuracy'
        
    def pipeline(self, threshold = None):
        est = make_pipeline(
            CountVectorizer(binary=False, analyzer='word', stop_words='english', max_features = 200000, min_df = 0.00001, ngram_range = (1, 3)),
            TfidfTransformer(norm = 'l2', smooth_idf = True, use_idf = True),
            self.algorithm,
        )
        return est

    def train_model(self, X, y):
        # X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify = y)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y)
        cv_strategy = StratifiedShuffleSplit(n_splits = self.nfolds, test_size = 0.3)
        # cv_strategy = KFold(3)

        estimator = self.pipeline()
        search = GridSearchCV(estimator, self.grid_pars, cv = cv_strategy, verbose = 10, n_jobs = self.n_jobs, scoring = self.scorer)
        search.fit(X_tr, y_tr)

        est = search.best_estimator_
        train_predictions = est.predict(X_tr)
        test_predictions = est.predict(X_te)

        print()
        print()
        print('Best Parameters', search.best_params_)

        joblib.dump(search, self.name + self.ofile_extension, compress = 1)
        self.show_quality(search, y_tr, train_predictions, y_te, test_predictions)

    def show_quality(self, search, y_tr, train_predictions, y_te, test_predictions):
        print('On test sample accuracy:', metrics.accuracy_score(y_te, test_predictions))
        print('On training sample accuracy:', metrics.accuracy_score(y_tr, train_predictions))
        # print(metrics.classification_report(y_tr, train_predictions, y_tr))

    # TODO: Finish this
    def restore_from_file(self):
        pass


class MultiLabelTrainingBundle(TrainingBundle):

    ofile_extension = '-multiple.pkl'
    def __init__(self, name, algorithm, grid_pars, nfolds, n_jobs = 40):
        super(MultiLabelTrainingBundle, self).__init__(
            name, None, grid_pars, nfolds, n_jobs)
        self.algorithm = OneVsRestClassifier(algorithm)
        self.scorer = make_scorer(f1_score, average = 'micro')
        self.mlb = MultiLabelBinarizer()

    def transform_labels(self, y):
        return self.mlb.fit_transform(y)

    # TODO: Train on most popular labels
    #       then compare output
    #
    def train_model(self, X, y):
        y = self.mlb.fit_transform(y)
        return super(MultiLabelTrainingBundle, self).train_model(X, y)
  
    def show_quality(self, search, y_tr, train_predictions, y_te, test_predictions):
        print('On test sample accuracy:', metrics.f1_score(y_te, test_predictions, average = 'micro'))
        print()
        print('On training sample accuracy:', metrics.f1_score(y_tr, train_predictions, average = 'micro'))
        # print(metrics.classification_report(y_tr, train_predictions, y_tr))

        print()
        y_te, test_predictions = self.mlb.inverse_transform(y_te), self.mlb.inverse_transform(test_predictions)
        print('Using custom F1 measure', f1_mean(y_te, test_predictions))

        for true, test in zip(y_te[0:10], test_predictions[0:10]):
            print('True {0} predicted {1}'.format(true, test))
