from chirp_python.data_source import DataSource
from chirp_python.chdr import CHDR

class Scorer:
    voter_error = [0.0] * 30
    voter = 0
    num_nn = 0
    count = 0

    def __init__(self, n_pts):
        self.remaining_points = n_pts

    def score_training_data(self, training_data: DataSource, chdrs, n_vars):
        from chirp_python.classifier import Classifier
        num_scored = 0
        error_rate = 0
        starting_points = self.remaining_points
        row = [0.0] * n_vars
        for i in range(len(training_data.predicted_values)):
            if training_data.predicted_values[i] < 0:
                for j in range(n_vars):
                    row[j] = training_data.data[i][j]
                predicted = -1
                for ch in chdrs:
                    predicted = ch.score(row)
                    if predicted >= 0:
                        self.remaining_points -= 1
                        num_scored += 1
                        break
                training_data.predicted_values[i] = predicted
                observed = training_data.class_values[i]
                if observed != predicted:
                    error_rate += 1
        
        if len(training_data.predicted_values) > 0:
            error_rate /= len(training_data.predicted_values)
        
        if num_scored > 0 and Classifier.debug:
            print(f"Error = {error_rate}, scored = {num_scored}, remaining = {self.remaining_points}")
        
        if starting_points > 0:
            return num_scored / starting_points
        return 0

    def score_testing_data(self, instance, testing_data: DataSource, chdrs, n_vars):
        if not chdrs:
            return None
        row = [0.0] * n_vars
        
        for j in range(n_vars):
            row[j] = instance[j] # instance.value(j) in weka
            
        predicted = -1
        weight = 10
        for chdr in chdrs:
            predicted = chdr.score(row)
            if predicted >= 0:
                break
        
        if predicted < 0:
            predicted = self._score_using_nearest_rectangle(row, chdrs)
            Scorer.num_nn += 1
            weight = 3
            
        Scorer.count += 1
        return [predicted, weight]

    def _score_using_nearest_rectangle(self, row, chdrs):
        min_dist = float('inf')
        min_k = -1
        for k, chdr in enumerate(chdrs):
            dist = chdr.find_closest_rect(row)
            if dist < min_dist:
                min_dist = dist
                min_k = k
        
        chdr = chdrs[min_k]
        return chdr.class_index

    def delete_points(self, data_source: DataSource, chdr: CHDR, n_vars):
        row = [0.0] * n_vars
        for i in range(data_source.n_pts):
            if data_source.predicted_values[i] < 0:
                for j in range(n_vars):
                    row[j] = data_source.data[i][j]
                data_source.predicted_values[i] = chdr.score(row)

