import numpy as np
from chirp_python.hdr import HDR

class CHDR:
    def __init__(self, class_index, class_name, n_classes, n_cats, n_bins,
                 xwt, ywt, x_bounds, y_bounds):
        self.class_index = class_index
        self.class_name = class_name
        self.n_classes = n_classes
        self.n_cats = n_cats
        self.n_bins = n_bins
        self.xwt = xwt
        self.ywt = ywt
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.hdrs = []

    def add(self, hdr: HDR):
        self.hdrs.append(hdr)

    def score(self, row):
        x_val = self.get_x(row)
        y_val = self.get_y(row)
        for hdr in self.hdrs:
            if hdr.contains_point(x_val, y_val):
                return self.class_index
        return -1

    def find_closest_rect(self, row):
        x_val = self.get_x(row)
        y_val = self.get_y(row)
        dist = float('inf')
        for hdr in self.hdrs:
            dist = min(hdr.rect_infinity_distance(x_val, y_val), dist)
        return dist

    def get_x(self, row):
        return (self._compute_projection(row, self.xwt) - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0])

    def get_y(self, row):
        return (self._compute_projection(row, self.ywt) - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0])

    def _compute_projection(self, row, wi):
        nwt = 0
        x = 0
        for j in range(len(wi)):
            wt = 1.0
            if wi[j] < 0:
                wt = -1.0
            wij = abs(wi[j])
            xi = row[wij]
            if wij < self.n_cats:
                xi = 0
            if not np.isnan(xi):
                x += wt * xi
                nwt += 1
        if nwt > 0 and nwt < len(wi):
            x = x * len(wi) / nwt
        return x
