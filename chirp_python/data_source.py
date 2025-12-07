import numpy as np

class DataSource:
    FUZZ0 = 0.00001
    FUZZ1 = 0.99999

    def __init__(self, data=None, class_values=None, class_names=None):
        from chirp_python.data_transforms import DataTransforms
        if data is not None:
            self.data = data
            self.class_values = class_values
            self.class_names = class_names if class_names is not None else []
            
            # Data is expected as (samples, features)
            self.n_pts = data.shape[0]
            self.n_vars = data.shape[1]
            self.n_cons = self.n_vars
            
            self.n_classes = len(self.class_names) if self.class_names else 0
            if self.n_classes == 0 and self.class_values is not None:
                self.n_classes = np.max(self.class_values) + 1

            self.predicted_values = np.full(self.n_pts, -1, dtype=int)
            self.transforms = DataTransforms(self)
            self.class_means = None
            self.class_counts = None
            self.category_scores = None
            
            # Automatically compute statistics upon initialization
            self._compute_class_statistics()

        else:
            # Initialize with empty/default values
            self.data = None
            self.class_values = None
            self.class_names = []
            self.n_vars = 0
            self.n_cons = 0
            self.n_pts = 0
            self.n_classes = 0
            self.predicted_values = None
            self.transforms = None
            self.class_means = None
            self.class_counts = None
            self.category_scores = None

    def set_transforms(self, transforms):
        self.transforms = transforms

    def transform_data(self):
        self.transforms.transform_data()

    def transform_testing_instance(self, instance):
        self.transforms.transform_testing_instance(instance)

    def update_class_statistics(self):
        self._compute_class_statistics()

    def _compute_class_statistics(self):
        if self.data is None or self.class_values is None or self.n_classes == 0:
            return
            
        self.class_means = np.zeros((self.n_classes, self.n_vars))
        self.class_counts = np.zeros(self.n_classes, dtype=int)
        
        for i in range(self.n_pts):
            class_idx = self.class_values[i]
            if self.predicted_values[i] < 0 and class_idx >= 0:
                self._update(class_idx, i)
        
        # To avoid division by zero for classes with no samples
        safe_counts = np.where(self.class_counts <= 0, 1, self.class_counts)
        for i in range(self.n_classes):
             self.class_means[i] /= safe_counts[i]


    def _update(self, class_idx, row_index):
        valid_data = self.data[row_index, :]
        
        # Using boolean indexing to handle NaNs correctly
        is_not_nan = ~np.isnan(valid_data)
        
        # Incrementally update means
        self.class_means[class_idx, is_not_nan] += valid_data[is_not_nan]
        self.class_counts[class_idx] += 1
