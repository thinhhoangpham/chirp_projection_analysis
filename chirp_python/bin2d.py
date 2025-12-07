from chirp_python.binner import Binner

class Bin2D:
    """simple container class with public fields to keep accessors simple"""
    def __init__(self, binner: Binner):
        self.binner = binner
        self.chdr = binner.get_chdr()
        self.pure_count = 0
        self._compute_pure_count()

    def _compute_pure_count(self):
        self.pure_count = 0
        n_bins = self.binner.get_num_bins()
        for i in range(n_bins):
            for j in range(n_bins):
                self.pure_count += self.binner.pure_count(i, j)
