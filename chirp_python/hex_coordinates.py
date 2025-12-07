import numpy as np
import math

class HexCoordinates:
    """Utility class for hexagonal coordinate system operations"""
    
    @staticmethod
    def cartesian_to_hex(x, y, hex_size):
        """
        Convert Cartesian coordinates to hexagonal axial coordinates (q, r)
        Using flat-top hexagon orientation
        
        Args:
            x, y: Cartesian coordinates in [0, 1] space
            hex_size: Size of hexagons (distance from center to edge)
        
        Returns:
            (q, r): Axial coordinates
        """
        # Scale coordinates to hex grid
        # For flat-top hexagons
        sqrt3 = math.sqrt(3)
        
        # Convert to hex coordinates
        q = (2.0/3.0 * x) / hex_size
        r = (-1.0/3.0 * x + sqrt3/3.0 * y) / hex_size
        
        return HexCoordinates.hex_round(q, r)
    
    @staticmethod
    def hex_to_cartesian(q, r, hex_size):
        """
        Convert hexagonal axial coordinates to Cartesian coordinates
        
        Args:
            q, r: Axial coordinates  
            hex_size: Size of hexagons
            
        Returns:
            (x, y): Cartesian coordinates
        """
        sqrt3 = math.sqrt(3)
        x = hex_size * (3.0/2.0 * q)
        y = hex_size * (sqrt3/2.0 * q + sqrt3 * r)
        return x, y
    
    @staticmethod
    def hex_round(q, r):
        """
        Round fractional hex coordinates to nearest integer hex coordinates
        
        Args:
            q, r: Fractional axial coordinates
            
        Returns:
            (q_int, r_int): Integer axial coordinates
        """
        # Convert to cube coordinates for easier rounding
        s = -q - r
        
        rq = round(q)
        rr = round(r)
        rs = round(s)
        
        q_diff = abs(rq - q)
        r_diff = abs(rr - r)
        s_diff = abs(rs - s)
        
        if q_diff > r_diff and q_diff > s_diff:
            rq = -rr - rs
        elif r_diff > s_diff:
            rr = -rq - rs
        
        return int(rq), int(rr)
    
    @staticmethod
    def get_hex_bounds(n_rings):
        """
        Get the bounds for a hexagonal grid with n_rings
        
        Args:
            n_rings: Number of rings around center (radius)
            
        Returns:
            List of (q, r) coordinates for all hexagons in the grid
        """
        hexagons = []
        for q in range(-n_rings, n_rings + 1):
            r1 = max(-n_rings, -q - n_rings)
            r2 = min(n_rings, -q + n_rings)
            for r in range(r1, r2 + 1):
                hexagons.append((q, r))
        return hexagons
    
    @staticmethod
    def hex_distance(q1, r1, q2, r2):
        """
        Calculate distance between two hexagons in axial coordinates
        
        Args:
            q1, r1: First hexagon coordinates
            q2, r2: Second hexagon coordinates
            
        Returns:
            Distance (number of hex steps)
        """
        return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2
    
    @staticmethod
    def calculate_hex_size_for_grid(n_rings):
        """
        Calculate appropriate hex size to fit n_rings in [0,1]x[0,1] space
        
        Args:
            n_rings: Number of rings in the hexagonal grid
            
        Returns:
            hex_size: Size parameter for hexagons
        """
        # For flat-top hexagons, the width is 2*hex_size and height is sqrt(3)*hex_size
        # We need to fit (2*n_rings + 1) hexagons in each direction
        grid_width = 2 * n_rings + 1
        grid_height = 2 * n_rings + 1
        
        # Calculate size based on fitting in [0,1] space
        hex_size_x = 1.0 / (1.5 * grid_width)
        hex_size_y = 1.0 / (math.sqrt(3) * grid_height)
        
        return min(hex_size_x, hex_size_y)
    
    @staticmethod
    def get_neighbors(q, r):
        """
        Get the 6 neighboring hexagon coordinates
        
        Args:
            q, r: Hexagon coordinates
            
        Returns:
            List of (q, r) coordinates for neighbors
        """
        # Axial coordinate directions for flat-top hexagons
        directions = [
            (1, 0), (1, -1), (0, -1),
            (-1, 0), (-1, 1), (0, 1)
        ]
        
        neighbors = []
        for dq, dr in directions:
            neighbors.append((q + dq, r + dr))
        
        return neighbors
