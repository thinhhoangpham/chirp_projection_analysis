import numpy as np
from chirp_python.data_source import DataSource
from chirp_python.projection_full import Projection as Projection75


class SimpleProjection(Projection75):
    """Projection that uses best variables with 75% sparsity but without complex optimization."""
    
    def simple_projection(self, ds: DataSource, current_class):
        """Generate a single projection using best variables with 75% sparsity.
        
        Args:
            ds: DataSource object
            current_class: Current class for optimization
        
        Returns:
            Weight vector with optimized signs using 75% of best variables
        """
        # Use the select_best_variables from base class
        best_vars = self.select_best_variables(ds, current_class)
        if best_vars is None or len(best_vars) == 0:
            return None
            
        # Use 75% sparsity (same as projection_full.py)
        n_wts = max(3 * len(best_vars) // 4, 1)
        wi = self._generate_weights(ds, best_vars, current_class, n_wts)
        return wi
    
    def simple_projection_no_sign_opt(self, ds: DataSource, current_class):
        """Generate a projection without sign optimization (for later manual flipping).
        
        Args:
            ds: DataSource object
            current_class: Current class for optimization
        
        Returns:
            Weight vector with all positive signs (before optimization)
        """
        # Use the select_best_variables from base class
        best_vars = self.select_best_variables(ds, current_class)
        if best_vars is None or len(best_vars) == 0:
            return None
            
        # Use 75% sparsity
        n_wts = max(3 * len(best_vars) // 4, 1)
        
        # Generate weights without sign optimization
        wi = np.zeros(n_wts, dtype=int)
        if len(best_vars) < self.m_wts:
            self.random.shuffle(best_vars)
        np.copyto(wi, best_vars[:n_wts])
        
        # Don't call _set_negatives - keep all positive
        return wi
