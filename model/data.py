import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

class DataHandler:

    @classmethod
    def load_data(klass, datafile='data/categories.csv' ,catfile='data/category_mapping.csv'):
        df = pd.read_csv(datafile)
        df = klass.clean(df)
        df = klass.popular(df)
        cat_map = klass.cat_mapping(catfile)
        return klass.useful_features(df, cat_map)

    @classmethod
    def clean(klass, df):
        df = df.assign(categories=lambda x: x.categories.map(eval))
        df = df.dropna()
        df = df.loc[lambda x: x.categories.map(len) > 0]
        return df

    @classmethod
    def popular(klass, df):
        category_dict = [{cat: 1 for cat in cats} for cats in list(df.categories)]
        encoder = DictVectorizer()
        encoder.fit(category_dict)
        categories_mat = encoder.transform(category_dict)
        categories_counts = np.asarray(categories_mat.sum(0).reshape(-1), dtype=np.int)[0]
        categories_dist = dict(zip(encoder.feature_names_, categories_counts))
        df['most_popular_category'] = df.categories.map(
            lambda x: sorted(x, key=lambda x: categories_dist[x], reverse=True)[0])
        return df

    @classmethod
    def cat_mapping(klass, catfile):
        df_category_mapping = pd.read_csv(catfile, sep='\t')
        return  dict(zip(df_category_mapping.raw, df_category_mapping.mapped)) 

    @classmethod
    def useful_features(klass, df, category_mapping):
        descriptions = df['short_description']
        categories = df['categories'].map(lambda x: [category_mapping[c] for c in x])
        return descriptions.values, categories.values

    @staticmethod
    def flatten_data(X, Y):
        X_flat = []
        y_flat = []
        for x, ys in zip(X, Y):
            for y in ys:
                X_flat.append(x)
                y_flat.append(y)
        return X_flat, y_flat


class DataHandlerSingle(DataHandler):
    def __init__(self):
        super(DataHandlerSingle, self).__init__()

    def useful_features(df, category_mapping):
        descriptions = df['short_description']
        categories = df['most_popular_category'].map(lambda x: category_mapping[x])
        return descriptions.values, categories.values


