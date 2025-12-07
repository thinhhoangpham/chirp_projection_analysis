import numpy as np

class HexBin:
    """Hexagonal bin container class with public fields to keep accessors simple"""
    def __init__(self, n_classes):
        self.class_counts = np.zeros(n_classes, dtype=int)
        self.centroid = np.zeros(2)
        self.count = 0
        self.is_covered = False
        # Hexagonal coordinates (q, r) in axial coordinate system
        self.q = 0
        self.r = 0
        # Cartesian center coordinates for this hex
        self.hex_center_x = 0.0
        self.hex_center_y = 0.0

    def is_pure(self, class_index, purity_threshold):
        if self.count == 0:
            return False
        return float(self.class_counts[class_index]) / self.count >= purity_threshold

    def set_hex_coordinates(self, q, r, hex_center_x, hex_center_y):
        """Set the hexagonal grid coordinates and center position"""
        self.q = q
        self.r = r
        self.hex_center_x = hex_center_x
        self.hex_center_y = hex_center_y
