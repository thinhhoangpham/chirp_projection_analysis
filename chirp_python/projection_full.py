import numpy as np
from chirp_python.projection import Projection as BaseProjection
from chirp_python.data_source import DataSource


class Projection(BaseProjection):
    """Projection variant that uses 75% sparsity (top 3/4 of selected variables)."""

    def good_projection(self, weight_indices, ds: DataSource, current_class):
        # Use 75% of selected variables (match Java: 3 * len / 4, floored)
        n_wts = max(3 * len(weight_indices) // 4, 1)
        wi = self._generate_weights(ds, weight_indices, current_class, n_wts)
        return wi

    def good_projection_with_logging(self, weight_indices, ds: DataSource, current_class):
        # Use 75% of selected variables with a single logging entry
        logging_data = []
        n_wts = max(3 * len(weight_indices) // 4, 1)

        wi, sparsity_log = self._generate_weights_with_logging(ds, weight_indices, current_class, n_wts)
        dist = self._composite_absolute_difference(ds, wi, current_class)

        sparsity_log['final_quality'] = dist
        sparsity_log['sparsity_percent'] = 75
        sparsity_log['n_variables'] = n_wts
        logging_data.append(sparsity_log)

        return wi, logging_data

    # Override sign-flipping rounds to 10 (instead of base class default 5)
    def _set_negatives(self, ds, wi, current_class):
        temperature = 0.01
        for _ in range(10):
            for _ in range(max(1, len(wi) // 2)):
                pre_move = self._composite_absolute_difference(ds, wi, current_class)
                self._move(wi)
                post_move = self._composite_absolute_difference(ds, wi, current_class)
                if not self._accept_move(pre_move - post_move, temperature):
                    self._reset(wi)
            temperature *= 0.9

    def _set_negatives_with_logging(self, ds, wi, current_class):
        """Set negatives with detailed logging of each sign flipping round (10 rounds)."""
        temperature = 0.01
        rounds_log = []

        for round_num in range(10):
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

