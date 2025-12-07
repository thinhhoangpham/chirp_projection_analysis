import numpy as np

class Sorter:
    @staticmethod
    def ascending_sort(x):
        return np.argsort(x)

    @staticmethod
    def descending_sort(x):
        return np.argsort(x)[::-1]
