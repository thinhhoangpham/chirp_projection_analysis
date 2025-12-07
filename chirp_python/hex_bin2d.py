from chirp_python.hex_binner import HexBinner

class HexBin2D:
    """Hexagonal 2D bin container class similar to Bin2D but for hexagonal binning"""
    def __init__(self, hex_binner: HexBinner):
        self.hex_binner = hex_binner
        self.chdr = hex_binner.get_chdr()
        self.pure_count = 0
        self._compute_pure_count()

    def _compute_pure_count(self):
        """Compute the total pure count across all hexagonal bins"""
        self.pure_count = 0
        
        # Iterate through all hexagonal coordinates
        for q, r in self.hex_binner.get_hex_coords():
            self.pure_count += self.hex_binner.pure_count(q, r)

    def get_binner(self):
        """Get the underlying hexagonal binner"""
        return self.hex_binner

    def get_pure_count(self):
        """Get the total pure count"""
        return self.pure_count

    def get_n_hexagons(self):
        """Get total number of hexagons in the grid"""
        return self.hex_binner.get_n_hexagons()

    def get_n_rings(self):
        """Get number of rings in the hexagonal grid"""
        return self.hex_binner.get_n_rings()

    def get_occupied_hexagons(self):
        """Get list of occupied hexagon coordinates and their bins"""
        return self.hex_binner.get_occupied_bins()

    def get_hex_stats(self):
        """
        Get statistics about the hexagonal binning
        
        Returns:
            dict: Statistics including total hexagons, occupied hexagons, pure hexagons, etc.
        """
        total_hexagons = self.get_n_hexagons()
        occupied_bins = self.get_occupied_hexagons()
        occupied_count = len(occupied_bins)
        
        pure_hexagons = 0
        total_points_in_pure_hexagons = 0
        
        for coords, bin_obj in occupied_bins:
            if self.hex_binner.pure_count(coords[0], coords[1]) > 0:
                pure_hexagons += 1
                total_points_in_pure_hexagons += self.hex_binner.pure_count(coords[0], coords[1])
        
        stats = {
            'total_hexagons': total_hexagons,
            'occupied_hexagons': occupied_count,
            'pure_hexagons': pure_hexagons,
            'pure_count': self.pure_count,
            'total_points_in_pure_hexagons': total_points_in_pure_hexagons,
            'occupancy_rate': occupied_count / total_hexagons if total_hexagons > 0 else 0,
            'purity_rate': pure_hexagons / occupied_count if occupied_count > 0 else 0,
            'n_rings': self.get_n_rings()
        }
        
        return stats
