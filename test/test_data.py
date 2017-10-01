import unittest
from collections import Counter

import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

from model.data import DataHandlerSingle


def integral(data, threshold = 0):
    return sum(map(lambda x, y, _:  x * y * (y > threshold), *data))


class TestData(unittest.TestCase):


    def test_data(self):
        X, y = DataHandlerSingle.load_data(threshold = 0)


        yy = sorted(Counter(y).values())
        data = plt.hist(yy, bins = 1000)
        plt.yscale('log')
        plt.xscale('log')
        plt.title('Probability of different tag frequencies')
        plt.xlabel('tag frequency')
        plt.ylabel('frequency of tag frequencies')



        plt.show()
        x, total = np.linspace(1, 1000, 10), integral(data)
        y = np.array(list(map(lambda t: integral(data, t), x)))
        plt.plot(x, y / total, 'o')
        plt.title('Data loss')
        plt.xlabel('threshold on frequency')
        plt.ylabel('data rate')
        plt.show()




 