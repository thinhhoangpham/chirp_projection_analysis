import numpy as np
from chirp_python.data_source import DataSource
from chirp_python.sorter import Sorter

class Projection:
    def __init__(self, random, m_wts):
        self.random = random
        self.m_wts = m_wts
        self.ran_loc = 0

    def good_projection(self, weight_indices, ds: DataSource, current_class):
        bi = None
        best = -1
        for p in range(1, 4):
            n_wts = max(p * len(weight_indices) // 4, 1)
            wi = self._generate_weights(ds, weight_indices, current_class, n_wts)
            dist = self._composite_absolute_difference(ds, wi, current_class)
            if dist > best:
                best = dist
                bi = wi
        return bi
        
    def good_projection_with_logging(self, weight_indices, ds: DataSource, current_class):
        """Version that returns detailed logging information"""
        bi = None
        best = -1
        logging_data = []
        
        for p in range(1, 4):
            sparsity_percent = p * 25
            n_wts = max(p * len(weight_indices) // 4, 1)
            
            # Generate weights with logging
            wi, sparsity_log = self._generate_weights_with_logging(ds, weight_indices, current_class, n_wts)
            dist = self._composite_absolute_difference(ds, wi, current_class)
            
            sparsity_log['final_quality'] = dist
            sparsity_log['sparsity_percent'] = sparsity_percent
            sparsity_log['n_variables'] = n_wts
            logging_data.append(sparsity_log)
            
            if dist > best:
                best = dist
                bi = wi
                
        return bi, logging_data

    def select_best_variables(self, ds: DataSource, current_class):
        if ds.class_means is None:
            return None
        n_variables = ds.class_means.shape[1]
        if n_variables < self.m_wts:
            var_indices = np.arange(n_variables)
        else:
            distances = np.zeros(n_variables)
            for i in range(n_variables):
                distances[i] = self._univariate_absolute_difference(ds, i, current_class)
            indices = Sorter.descending_sort(distances)
            var_indices = indices[:self.m_wts]
        return var_indices

    def _univariate_absolute_difference(self, ds: DataSource, variable, current_class):
        min_difference = float('inf')
        for k in range(len(ds.class_means)):
            if k != current_class:
                difference = abs(ds.class_means[current_class][variable] - ds.class_means[k][variable])
                min_difference = min(difference, min_difference)
        return min_difference

    def _composite_absolute_difference(self, ds: DataSource, wi, current_class):
        current_centroid = 0
        n = 0
        for j in range(len(wi)):
            wt = 1.
            if wi[j] < 0:
                wt = -1.
            if not np.isnan(ds.class_means[current_class][abs(wi[j])]):
                current_centroid += wt * ds.class_means[current_class][abs(wi[j])]
                n += 1
        if n > 0:
            current_centroid /= n
        
        min_dist = float('inf')
        for k in range(len(ds.class_means)):
            if k != current_class:
                other_centroid = 0
                n = 0
                for j in range(len(wi)):
                    wt = 1.
                    if wi[j] < 0:
                        wt = -1.
                    if not np.isnan(ds.class_means[k][abs(wi[j])]):
                        other_centroid += wt * ds.class_means[k][abs(wi[j])]
                        n += 1
                if n > 0:
                    other_centroid /= n
                difference = abs(other_centroid - current_centroid)
                min_dist = min(difference, min_dist)
        return min_dist

    def _generate_weights(self, ds: DataSource, weight_indices, current_class, n_wts):
        wi = np.zeros(n_wts, dtype=int)
        if len(weight_indices) < self.m_wts:
            self.random.shuffle(weight_indices)
        np.copyto(wi, weight_indices[:n_wts])
        self._set_negatives(ds, wi, current_class)
        return wi
        
    def _generate_weights_with_logging(self, ds: DataSource, weight_indices, current_class, n_wts):
        """Generate weights with detailed logging of sign flipping process"""
        wi = np.zeros(n_wts, dtype=int)
        if len(weight_indices) < self.m_wts:
            self.random.shuffle(weight_indices)
        np.copyto(wi, weight_indices[:n_wts])
        
        # Get initial state
        initial_weights = wi.copy()
        initial_quality = self._composite_absolute_difference(ds, wi, current_class)
        
        # Perform sign flipping with logging
        sign_flipping_log = self._set_negatives_with_logging(ds, wi, current_class)
        
        # Compile logging data
        logging_data = {
            'initial_weights': initial_weights,
            'initial_quality': initial_quality,
            'final_weights': wi.copy(),
            'sign_flipping_rounds': sign_flipping_log
        }
        
        return wi, logging_data

    def _set_negatives(self, ds: DataSource, wi, current_class):
        temperature = 0.01
        for _ in range(5):
            for _ in range(max(1, len(wi) // 2)):
                pre_move = self._composite_absolute_difference(ds, wi, current_class)
                self._move(wi)
                post_move = self._composite_absolute_difference(ds, wi, current_class)
                if not self._accept_move(pre_move - post_move, temperature):
                    self._reset(wi)
            temperature *= 0.9
            
    def _set_negatives_with_logging(self, ds: DataSource, wi, current_class):
        """Set negatives with detailed logging of each sign flipping round"""
        temperature = 0.01
        rounds_log = []
        
        for round_num in range(5):
            round_log = {
                'round': round_num + 1,
                'temperature': temperature,
                'moves': []
            }
            
            for move_num in range(max(1, len(wi) // 2)):
                # Log state before move
                pre_move_weights = wi.copy()
                pre_move_quality = self._composite_absolute_difference(ds, wi, current_class)
                
                # Perform move
                self._move(wi)
                post_move_quality = self._composite_absolute_difference(ds, wi, current_class)
                
                # Calculate improvement
                improvement = post_move_quality - pre_move_quality
                
                # Decide whether to accept
                accepted = self._accept_move(-improvement, temperature)
                
                move_log = {
                    'move': move_num + 1,
                    'pre_move_weights': pre_move_weights.copy(),
                    'pre_move_quality': pre_move_quality,
                    'post_move_weights': wi.copy(),
                    'post_move_quality': post_move_quality,
                    'improvement': improvement,
                    'accepted': accepted,
                    'flipped_position': self.ran_loc
                }
                
                if not accepted:
                    # Reset the move
                    self._reset(wi)
                    move_log['post_move_weights'] = wi.copy()
                    move_log['post_move_quality'] = pre_move_quality
                
                round_log['moves'].append(move_log)
            
            # Add round summary
            accepted_moves = sum(1 for move in round_log['moves'] if move['accepted'])
            round_log['moves_attempted'] = len(round_log['moves'])
            round_log['moves_accepted'] = accepted_moves
            round_log['acceptance_rate'] = accepted_moves / len(round_log['moves']) if len(round_log['moves']) > 0 else 0.0
            round_log['final_quality'] = self._composite_absolute_difference(ds, wi, current_class)
            
            rounds_log.append(round_log)
            temperature *= 0.9
            
        return rounds_log

    def _accept_move(self, delta, t):
        if delta < 0:
            return True
        else:
            return self.random.random() < np.exp(-delta / t)

    def _move(self, wi):
        if len(wi) == 1:
            self.ran_loc = 0
        else:
            self.ran_loc = self.random.randint(0, len(wi)-1)
        wi[self.ran_loc] = -wi[self.ran_loc]

    def _reset(self, wi):
        wi[self.ran_loc] = -wi[self.ran_loc]

