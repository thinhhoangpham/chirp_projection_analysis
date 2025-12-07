import numpy as np

# Inherit the exact 75% sparsity behavior, but add a separation threshold filter
from chirp_python.projection_full import Projection as Projection75Base
from chirp_python.data_source import DataSource


class Projection75Thresholded(Projection75Base):
    """Projection that enforces a univariate separation threshold before initialization.

    - Keeps 75% sparsity target from the parent implementation.
    - Before shuffling/initialization, filters the candidate feature indices to those with
      univariate separation ability > 0.0. If that yields fewer than the 75% target,
      it accepts the smaller set (as requested).
    - Falls back to the original indices if filtering removes everything.
    """

    def _filter_indices_by_separation(self, ds: DataSource, weight_indices, current_class: int, threshold: float = 0.0):
        # Preserve incoming order; include only features with separation > threshold
        filtered = []
        for idx in weight_indices:
            sep = self._univariate_absolute_difference(ds, int(idx), current_class)
            if sep > threshold:
                filtered.append(int(idx))
        return filtered

    def good_projection(self, weight_indices, ds: DataSource, current_class):
        # Apply threshold filter first
        filtered = self._filter_indices_by_separation(ds, weight_indices, current_class, threshold=0.0)
        if len(filtered) == 0:
            filtered = list(weight_indices)

        # Maintain 75% sparsity target, but accept fewer if filtering reduces the pool
        desired_n_wts = max(3 * len(weight_indices) // 4, 1)
        n_wts = max(1, min(desired_n_wts, len(filtered)))

        wi = self._generate_weights(ds, np.asarray(filtered, dtype=int), current_class, n_wts)
        return wi

    def good_projection_with_logging(self, weight_indices, ds: DataSource, current_class):
        logging_data = []

        # Apply threshold filter first
        filtered = self._filter_indices_by_separation(ds, weight_indices, current_class, threshold=0.0)
        if len(filtered) == 0:
            filtered = list(weight_indices)

        # Maintain 75% sparsity target, but accept fewer if filtering reduces the pool
        desired_n_wts = max(3 * len(weight_indices) // 4, 1)
        n_wts = max(1, min(desired_n_wts, len(filtered)))

        wi, sparsity_log = self._generate_weights_with_logging(ds, np.asarray(filtered, dtype=int), current_class, n_wts)
        dist = self._composite_absolute_difference(ds, wi, current_class)

        # Match the parent logging contract
        sparsity_log['final_quality'] = dist
        sparsity_log['sparsity_percent'] = 75
        sparsity_log['n_variables'] = n_wts
        logging_data.append(sparsity_log)

        return wi, logging_data


