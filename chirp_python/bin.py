import numpy as np

class Bin:
    """simple container class with public fields to keep accessors simple"""
    def __init__(self, n_classes):
        self.class_counts = np.zeros(n_classes, dtype=int)
        self.centroid = np.zeros(2)
        self.count = 0
        self.is_covered = False

    def is_pure(self, class_index, purity_threshold):
        if self.count == 0:
            return False
        return float(self.class_counts[class_index]) / self.count >= purity_threshold
