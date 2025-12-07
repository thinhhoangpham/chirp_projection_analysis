import numpy as np
from chirp_python.bin import Bin
from chirp_python.chdr import CHDR
from chirp_python.data_source import DataSource

class Binner:
    def __init__(self, chdr: CHDR):
        self.chdr = chdr
        self.n_bins = chdr.n_bins
        self.bins = [[Bin(chdr.n_classes) for _ in range(self.n_bins)] for _ in range(self.n_bins)]

    def pure_count(self, row, col):
        b = self.bins[row][col]
        if b.is_pure(self.chdr.class_index, 1.0):
            return b.class_counts[self.chdr.class_index]
        else:
            return 0

    def get_bins(self):
        return self.bins

    def get_num_bins(self):
        return self.n_bins

    def get_chdr(self):
        return self.chdr

    def compute(self, data: DataSource, x, y):
        class_values = data.class_values
        for j in range(self.n_bins):
            for k in range(self.n_bins):
                self.bins[j][k] = Bin(self.chdr.n_classes)
        
        for i in range(len(x)):
            if np.isnan(x[i]) or np.isnan(y[i]) or class_values[i] < 0:
                continue
            
            if data.predicted_values[i] < 0:
                ix = int(DataSource.FUZZ1 * x[i] * self.n_bins)
                iy = int(DataSource.FUZZ1 * y[i] * self.n_bins)
                ix = max(min(ix, self.n_bins - 1), 0)
                iy = max(min(iy, self.n_bins - 1), 0)
                
                b = self.bins[ix][iy]
                b.class_counts[class_values[i]] += 1
                b.count += 1
                b.centroid[0] += (x[i] - b.centroid[0]) / b.count
                b.centroid[1] += (y[i] - b.centroid[1]) / b.count
