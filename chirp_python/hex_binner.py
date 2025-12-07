import numpy as np
from chirp_python.hex_bin import HexBin
from chirp_python.hex_coordinates import HexCoordinates
from chirp_python.chdr import CHDR
from chirp_python.data_source import DataSource

class HexBinner:
    def __init__(self, chdr: CHDR, n_rings=5):
        self.chdr = chdr
        self.n_rings = n_rings
        
        # Get all hexagon coordinates for the grid
        self.hex_coords = HexCoordinates.get_hex_bounds(n_rings)
        self.n_hexagons = len(self.hex_coords)
        
        # Calculate hex size to fit in [0,1] space
        self.hex_size = HexCoordinates.calculate_hex_size_for_grid(n_rings)
        
        # Create bins dictionary using (q,r) as keys
        self.bins = {}
        for q, r in self.hex_coords:
            hex_bin = HexBin(chdr.n_classes)
            hex_center_x, hex_center_y = HexCoordinates.hex_to_cartesian(q, r, self.hex_size)
            hex_bin.set_hex_coordinates(q, r, hex_center_x, hex_center_y)
            self.bins[(q, r)] = hex_bin

    def pure_count(self, q, r):
        """Get pure count for hexagon at coordinates (q, r)"""
        if (q, r) not in self.bins:
            return 0
        
        bin_obj = self.bins[(q, r)]
        if bin_obj.is_pure(self.chdr.class_index, 1.0):
            return bin_obj.class_counts[self.chdr.class_index]
        else:
            return 0

    def get_bin(self, q, r):
        """Get bin at hexagonal coordinates (q, r)"""
        return self.bins.get((q, r), None)

    def get_all_bins(self):
        """Get all bins as a dictionary"""
        return self.bins

    def get_hex_coords(self):
        """Get list of all hexagon coordinates"""
        return self.hex_coords

    def get_n_rings(self):
        """Get number of rings in the hex grid"""
        return self.n_rings
    
    def get_n_hexagons(self):
        """Get total number of hexagons"""
        return self.n_hexagons

    def get_chdr(self):
        return self.chdr

    def compute(self, data: DataSource, x, y):
        """
        Compute hexagonal binning for the given data points
        
        Args:
            data: DataSource object containing class information
            x, y: Arrays of normalized coordinates [0,1]
        """
        class_values = data.class_values
        
        # Initialize all bins
        for (q, r), bin_obj in self.bins.items():
            self.bins[(q, r)] = HexBin(self.chdr.n_classes)
            hex_center_x, hex_center_y = HexCoordinates.hex_to_cartesian(q, r, self.hex_size)
            self.bins[(q, r)].set_hex_coordinates(q, r, hex_center_x, hex_center_y)
        
        # Assign points to hexagonal bins
        for i in range(len(x)):
            if np.isnan(x[i]) or np.isnan(y[i]) or class_values[i] < 0:
                continue
            
            if data.predicted_values[i] < 0:
                # Convert cartesian to hex coordinates
                # Scale coordinates to fit in the hexagonal grid
                scaled_x = (x[i] - 0.5) * 2  # Scale from [0,1] to [-1,1]
                scaled_y = (y[i] - 0.5) * 2  # Scale from [0,1] to [-1,1]
                
                q, r = HexCoordinates.cartesian_to_hex(scaled_x, scaled_y, self.hex_size)
                
                # Check if this hex coordinate is within our grid
                if (q, r) in self.bins:
                    bin_obj = self.bins[(q, r)]
                    bin_obj.class_counts[class_values[i]] += 1
                    bin_obj.count += 1
                    
                    # Update centroid incrementally (weighted average of actual point positions)
                    bin_obj.centroid[0] += (x[i] - bin_obj.centroid[0]) / bin_obj.count
                    bin_obj.centroid[1] += (y[i] - bin_obj.centroid[1]) / bin_obj.count

    def get_hex_centers_cartesian(self):
        """
        Get cartesian coordinates of all hex centers for visualization
        
        Returns:
            x_centers, y_centers: Arrays of hex center coordinates in [0,1] space
        """
        x_centers = []
        y_centers = []
        
        for q, r in self.hex_coords:
            hex_x, hex_y = HexCoordinates.hex_to_cartesian(q, r, self.hex_size)
            # Convert back to [0,1] space
            norm_x = (hex_x / 2.0) + 0.5
            norm_y = (hex_y / 2.0) + 0.5
            x_centers.append(norm_x)
            y_centers.append(norm_y)
            
        return np.array(x_centers), np.array(y_centers)

    def get_occupied_bins(self):
        """
        Get only the bins that contain data points
        
        Returns:
            List of ((q, r), bin_obj) tuples for non-empty bins
        """
        occupied = []
        for coords, bin_obj in self.bins.items():
            if bin_obj.count > 0:
                occupied.append((coords, bin_obj))
        return occupied
