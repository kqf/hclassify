import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from collections import Counter

class DataHandler:

    @classmethod
    def load_data(klass, datafile='data/categories.csv' ,catfile='data/category_mapping.csv', threshold = 500):
        df = pd.read_csv(datafile)
        df = klass.clean(df)
        df = klass.categories(df, catfile)
        df = klass.popular(df, threshold)
        return klass.useful_features(df)

    @classmethod
    def clean(klass, df):
        df = df.assign(categories=lambda x: x.categories.map(eval))
        df = df.dropna()
        df = df.loc[lambda x: x.categories.map(len) > 0]
        return df

    @classmethod
    def categories(klass, df, catfile):
        df_category_mapping = pd.read_csv(catfile, sep='\t')
        category_mapping = dict(zip(df_category_mapping.raw, df_category_mapping.mapped))
        df['categories'] = df['categories'].map(lambda x: [category_mapping[c] for c in x])
        return df

    @classmethod
    def popular(klass, df, threshold):
        categories = []
        for x in list(df.categories):
            for c in set(x):
                categories.append(c)
        categories_dist = Counter(categories)

        def most_popular(x):
            max_cat = sorted(x, key=lambda y: categories_dist[y], reverse=True)[0]
            return max_cat
        
        df['most_popular_category'] = df.categories.map(most_popular)
        return df

    @classmethod
    def useful_features(klass, df):
        descriptions = df['short_description']
        return descriptions.values, df.categories.values

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

    @classmethod
    def popular(klass, df, threshold):
        df = super(DataHandlerSingle, klass).popular(df, threshold)
        categories_dist = Counter(df.most_popular_category)
        df.most_popular_category = df.most_popular_category.map(lambda x: x if categories_dist[x] > threshold else np.nan)
        df = df.dropna()
        return df

    def useful_features(df):
        descriptions = df['short_description']
        return descriptions.values, df.most_popular_category.values


