import numpy as np

class DataMoments:
    """
    A data structure representing the central moments of a distribution including:
    - the mean
    - the variance
    - the skewness
    - the kurtosis
    Data values are each passed into this data structure via the accumulate(...) method
    and the corresponding central moments are updated on each call
    """
    def __init__(self):
        self.moments = np.zeros(5)

    def accumulate(self, x):
        if np.isnan(x) or np.isinf(x):
            return
        n = self.moments[0]
        n1 = n + 1
        n2 = n * n
        delta = (self.moments[1] - x) / n1
        d2 = delta * delta
        d3 = delta * d2
        r1 = n / n1
        self.moments[4] += 4 * delta * self.moments[3] + 6 * d2 * self.moments[2] + (1 + n * n2) * d2 * d2
        self.moments[4] *= r1
        self.moments[3] += 3 * delta * self.moments[2] + (1 - n2) * d3
        self.moments[3] *= r1
        self.moments[2] += (1 + n) * d2
        self.moments[2] *= r1
        self.moments[1] -= delta
        self.moments[0] = n1

    def mean(self):
        return self.moments[1]

    def count(self):
        return self.moments[0]

    def kurtosis(self):
        if self.moments[0] < 4:
            return np.nan
        k_fact = (self.moments[0] - 2) * (self.moments[0] - 3)
        n1 = self.moments[0] - 1
        v = self.variance()
        return (self.moments[4] * self.moments[0] * self.moments[0] * (self.moments[0] + 1) / (v * v * n1) - n1 * n1 * 3) / k_fact

    def skewness(self):
        if self.moments[0] < 3:
            return np.nan
        v = self.variance()
        return self.moments[3] * self.moments[0] * self.moments[0] / (np.sqrt(v) * v * (self.moments[0] - 1) * (self.moments[0] - 2))

    def standard_deviation(self):
        return np.sqrt(self.variance())

    def variance(self):
        if self.moments[0] < 2:
            return np.nan
        return self.moments[2] * self.moments[0] / (self.moments[0] - 1)
