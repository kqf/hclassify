from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TrainingBundle():
    ofile_extension = '-single.pkl'
    def __init__(self, name, algorithm, grid_pars, nfolds, n_jobs = 40):
        self.name = name
        self.algorithm = algorithm
        self.grid_pars = grid_pars
        self.nfolds = nfolds
        self.n_jobs = n_jobs
        
    def pipeline(self):
        est = make_pipeline(
            CountVectorizer(binary=False, analyzer='word', stop_words='english'),
            TfidfTransformer(),
            self.algorithm,
        )
        # print ('>>>>', est.get_params(False))
        return est

    def transform_labels(self, y):
        return y

    def train_model(self, X, y):
        # Transform labels if needed
        y = self.transform_labels(y)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y)
        cv_strategy = StratifiedShuffleSplit(n_splits = self.nfolds, test_size = 0.3)
        # cv_strategy = KFold(3)

        estimator = self.pipeline()
        search = GridSearchCV(estimator, self.grid_pars, cv = cv_strategy, verbose = 10, n_jobs = self.n_jobs)
        search.fit(X_tr, y_tr)

        est = search.best_estimator_
        train_predictions = est.predict(X_tr)
        test_predictions = est.predict(X_te)

        print('On test sample accuracy:', metrics.accuracy_score(y_te, test_predictions))
        print()
        print()
        print('On training sample accuracy:', metrics.accuracy_score(y_tr, train_predictions))
        # print(metrics.classification_report(y_tr, train_predictions, y_tr))

        print()
        print()
        print('Best Parameters', search.best_params_)
        print('Best estimator', search.best_estimator_)
        joblib.dump(search, self.name + self.ofile_extension, compress = 1)

    # TODO: Finish this
    def restore_from_file(self):
        pass


class MultiLabelTrainingBundle(TrainingBundle):

    ofile_extension = '-multiple.pkl'
    def __init__(self, name, algorithm, grid_pars, nfolds, n_jobs = 40):
        super(MultiLabelTrainingBundle, self).__init__(
            name, None, grid_pars, nfolds, n_jobs)
        self.algorithm = OneVsRestClassifier(algorithm)
        

    def transform_labels(self, y):
        mlb = MultiLabelBinarizer()
        return mlb.fit_transform(y)
