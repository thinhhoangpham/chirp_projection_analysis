import numpy as np
import math
from chirp_python.data_source import DataSource
from chirp_python.projection import Projection
from chirp_python.scorer import Scorer
from chirp_python.sorter import Sorter
from chirp_python.bin2d import Bin2D
from chirp_python.chdr import CHDR
from chirp_python.binner import Binner
from chirp_python.rectangular_cover import RectangularCover

class Trainer:
    def __init__(self, training_instances, random, n_reps):
        self.m_pairs = 25
        self.n_pairs = 5
        self.m_wts = 50
        self.chdrs = []
        
        self.n_vars = training_instances.shape[1] - 1
        self.n_pts = training_instances.shape[0]
        
        data = np.array([training_instances.iloc[:, j] for j in range(self.n_vars)])
        
        class_values = training_instances.iloc[:, -1].astype(int).values
        class_names = np.unique(class_values).astype(str)
        
        self.training_data = DataSource(data, class_values, class_names, self.n_vars, self.n_pts)
        self.training_data.transform_data()
        
        self.n_cats = 0
        self.n_classes = self.training_data.n_classes
        self.is_classified = np.zeros(self.n_classes, dtype=bool)
        
        self.projection = Projection(random, self.m_wts)
        self.scorer = Scorer(self.n_pts)
        self.training_data.update_class_statistics(0)
        
        self.current_class = self.n_classes
        self.min_covered_points = min(max(self.n_pts // 10, 1), 10)
        self.purity_threshold = 1.0
        
        self.n_failures = 0
        self.best_bin2d = 0
        self.best_cover = 0
        self.bin2d = None
        self.bin2d_list = []
        self.current_pair = 0
        self.purity_indices = []

    def classify(self):
        self.n_failures = 0
        self._next_class()
        while self.n_failures < 2 * self.n_classes:
            self._process_class()
        while self._there_is_an_unclassified_class():
            self._process_remaining_unclassified_classes()

    def _process_class(self):
        n_covered = self._cover_bins()
        if n_covered > self.best_cover:
            self.best_bin2d = self.purity_indices[self.current_pair]
            self.best_cover = n_covered
        
        if self.current_pair < self.n_pairs:
            self._next_pair()
        else:
            if self.best_cover > self.min_covered_points:
                bin2d = self.bin2d_list[self.best_bin2d]
                self.chdrs.append(bin2d.chdr)
                p = self.scorer.score_training_data(self.training_data, self.chdrs, self.n_vars)
                if p < 0.01:
                    self.purity_threshold = 0.75
                self.n_failures = 0
                self.is_classified[self.current_class] = True
            self._next_class()

    def _process_remaining_unclassified_classes(self):
        self.n_failures = 0
        while self.n_failures < 2 * self.n_classes:
            self._process_class()

    def _there_is_an_unclassified_class(self):
        self.min_covered_points -= 1
        if self.min_covered_points == 0:
            return False
        return not np.all(self.is_classified)

    def _cover_bins(self):
        if self.training_data.class_counts[self.current_class] == 0:
            return 0
        r = RectangularCover(self.bin2d, self.min_covered_points, self.purity_threshold)
        return r.compute()

    def _next_pair(self):
        self.current_pair += 1
        self._get_next_2d_bin()

    def _next_class(self):
        self.current_pair = 0
        self.best_cover = -1
        self.n_failures += 1
        self._increment_class()
        self.training_data.update_class_statistics(self.current_class)
        if self.training_data.class_counts[self.current_class] > 0:
            self._build_2d_bin_list()
            self._get_next_2d_bin()

    def _increment_class(self):
        self.current_class += 1
        if self.current_class >= self.n_classes:
            self.current_class = 0

    def _get_next_2d_bin(self):
        current_index = self.purity_indices[self.current_pair]
        self.bin2d = self.bin2d_list[current_index]

    def _build_2d_bin_list(self):
        pure_counts = np.zeros(self.m_pairs)
        self.bin2d_list = []
        best_vars = self.projection.select_best_variables(self.training_data, self.current_class)
        n_bins = int(max(2 * math.log(self.scorer.remaining_points) / math.log(2), 10))
        for i in range(self.m_pairs):
            b2d = self._build_2d_bin(best_vars, n_bins)
            self.bin2d_list.append(b2d)
            pure_counts[i] = b2d.pure_count
        self.purity_indices = Sorter.descending_sort(pure_counts)

    def _build_2d_bin(self, best_vars, n_bins):
        class_name = self.training_data.class_names[self.current_class]
        xwt = self.projection.good_projection(best_vars, self.training_data, self.current_class)
        ywt = self.projection.good_projection(best_vars, self.training_data, self.current_class)
        x_bounds = self._compute_bounds(self.training_data, xwt, self.n_pts)
        y_bounds = self._compute_bounds(self.training_data, ywt, self.n_pts)
        chdr = CHDR(self.current_class, class_name, self.n_classes, self.n_cats, n_bins,
                    xwt, ywt, x_bounds, y_bounds)
        binner = Binner(chdr)
        x = self._fill_array(xwt, x_bounds)
        y = self._fill_array(ywt, y_bounds)
        binner.compute(self.training_data, x, y)
        return Bin2D(binner)

    def _fill_array(self, wi, bounds):
        result = np.zeros(self.n_pts)
        wt = np.ones(len(wi))
        wt[wi < 0] = -1.0
        
        for i in range(self.n_pts):
            nwt = 0
            for j in range(len(wi)):
                wij = abs(wi[j])
                xi = self.training_data.data[wij, i]
                if wij < self.n_cats:
                    k = int(xi * len(self.training_data.category_scores[wij]))
                    xi = self.training_data.category_scores[wij][k]
                if not np.isnan(xi):
                    result[i] += wt[j] * xi
                    nwt += 1
            if nwt > 0 and nwt < len(wi):
                result[i] = result[i] * len(wi) / nwt
            result[i] = (result[i] - bounds[0]) / (bounds[1] - bounds[0])
        return result

    def _compute_bounds(self, data_source, wi, n_pts):
        bounds = np.array([float('inf'), float('-inf')])
        for i in range(n_pts):
            nwt = 0
            data = 0
            for j in range(len(wi)):
                wt = 1.0
                if wi[j] < 0:
                    wt = -1.0
                wij = abs(wi[j])
                xi = data_source.data[wij, i]
                if wij < self.n_cats:
                    k = int(xi * len(data_source.category_scores[wij]))
                    xi = data_source.category_scores[wij][k]
                if not np.isnan(xi):
                    data += wt * xi
                    nwt += 1
            if nwt > 0 and nwt < len(wi):
                data = data * len(wi) / nwt
            if not np.isnan(data):
                bounds[0] = min(bounds[0], data)
                bounds[1] = max(bounds[1], data)
        
        bounds[0] -= DataSource.FUZZ0 * (bounds[1] - bounds[0])
        bounds[1] += DataSource.FUZZ0 * (bounds[1] - bounds[0])
        return bounds

    def score(self, instance, rep):
        transforms = self.training_data.transforms
        testing_data = DataSource()
        if rep == 0:
            testing_data.set_transforms(transforms)
            testing_data.transform_testing_instance(instance)
        return self.scorer.score_testing_data(instance, testing_data, self.chdrs, self.n_vars)
