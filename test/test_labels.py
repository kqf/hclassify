import pickle
import unittest
import numpy as np

from collections import Counter
from model.data import DataHandler, DataHandlerStrickt

def clean_data(data, to_remove):
    #for remove in to_remove:
    data = tuple(map(lambda x: frozenset([i for i in x if i != to_remove]), data))
    data = tuple(filter(lambda x: len(x) > 0, data))
    return data

def depth(data, threshold = 10):
    data = map(lambda x: frozenset(x), data)
    counter = Counter(data)
    throwaway = [v for v in counter.values() if v < threshold]
    return len(counter) - len(throwaway)

def clean(data):
    cat_distribution = Counter([x for cat in data for x in cat])
    to_remove = sorted(cat_distribution.keys(), key = lambda x: cat_distribution[x], reverse=True)

    results = {}
    for i, word in enumerate(to_remove):
        results[word] = depth(clean_data(data, word))
        print(i * 1. / len(to_remove), word, results[word])

    return results


def minimize_pairs(data, words):
    results = {}
    for word1 in words[0:-1]:
        data1 = clean_data(data, word1) 
        for word2 in words[1:]:
            print('Processing ', word1, word2)
            results[frozenset((word1, word2))] = depth(clean_data(data1, word2))
    return results


def factor_all(data, words):
    results, processed = {}, ''
    for word in words:
        processed += ', ' + word
        data = clean_data(data, word) 
        results[processed] = depth(data)
        print('Processing', processed, results[processed])
    return results


class TestLabels(unittest.TestCase):


    @unittest.skip('not useful')
    def test_data(self):
        _, y = DataHandler.load_data(threshold = 0)

        results = clean(tuple(y))
        for r in results:
            print (r, results[r])

        with open('categories-labels.pkl', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def less_important(self):

        with open('categories-labels.pkl', 'rb') as handle:
            data = pickle.load(handle)

        keys = sorted(data, key = lambda x : data[x])

        for k in keys[0:100]:
            print(k, data[k])

        return keys[0:100]


    @unittest.skip('')
    def test_pairs(self):
        _, y = DataHandler.load_data(threshold = 0)

        inp = self.less_important()
        minn = minimize_pairs(tuple(y), inp)
        print (minn)


    def test_pairs(self):
        _, y = DataHandlerStrickt.load_data(threshold = 10)
        print('Size of the dataset', len(y))
        print('For start dataset', depth(y))

        inp = self.less_important()
        minn = factor_all(tuple(y), inp)
        keys = sorted(minn, key = lambda x : minn[x])

        for i in range(10):
            print()

        for k in keys:
            print (k, minn[k])
