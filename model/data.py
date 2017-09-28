import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

class DataHandler:

    @classmethod
    def load_data(klass, datafile='data/categories.csv' ,catfile='data/category_mapping.csv'):
        df = pd.read_csv(datafile)
        klass.clean(df)
        klass.popular(df)
        return klass.useful_features(df, catfile)

    @classmethod
    def clean(klass, df):
        df = df.assign(categories=lambda x: x.categories.map(eval))
        df = df.dropna()
        df = df.loc[lambda x: x.categories.map(len) > 0]

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

    @classmethod
    def useful_features(klass, df, catfile):
        df_category_mapping = pd.read_csv(catfile, sep='\t')
        category_mapping = dict(zip(df_category_mapping.raw, df_category_mapping.mapped)) 
        descriptions = df['short_description']
        categories = df['categories'].map(lambda x: [category_mapping[c] for c in x])
        return descriptions.values, categories.values


def load_data():
    df_category_mapping = pd.read_csv('data/category_mapping.csv', sep='\t')
    category_mapping = dict(zip(df_category_mapping.raw, df_category_mapping.mapped))
    df = (pd.read_csv('data/categories.csv')
          .assign(categories=lambda x: x.categories.map(eval))
          .dropna()
          .loc[lambda x: x.categories.map(len) > 0])
    category_dict = [{cat: 1 for cat in cats} for cats in list(df.categories)]
    encoder = DictVectorizer()
    encoder.fit(category_dict)
    categories_mat = encoder.transform(category_dict)
    categories_counts = np.asarray(categories_mat.sum(0).reshape(-1), dtype=np.int)[0]
    categories_dist = dict(zip(encoder.feature_names_, categories_counts))
    df['most_popular_category'] = df.categories.map(
        lambda x: sorted(x, key=lambda x: categories_dist[x], reverse=True)[0])

    descriptions = df['short_description']
    categories = df['categories'].map(lambda x: [category_mapping[c] for c in x])

    return descriptions.values, categories.values


def flatten_data(X, Y):
    X_flat = []
    y_flat = []
    for x, ys in zip(X, Y):
        for y in ys:
            X_flat.append(x)
            y_flat.append(y)
    return X_flat, y_flat
