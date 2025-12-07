"""
Export utilities for CHIRP projection analysis.

This module provides functions to export projection data and results to various
formats, including:
- Bin data export to JSON (for interactive visualizations)
- Per-point projection coordinates export to JSON
- Formatting utilities for projection feature descriptions
"""

import json
import os
import numpy as np
from typing import Dict, List, Any


def format_projection_features(weights: np.ndarray, transforms: List[str],
                               features: List[str]) -> List[str]:
    """Format projection features with transformations for display.

    Args:
        weights: Array of signed feature indices (sign indicates +/-)
        transforms: List of transformation types for each weight
        features: List of feature names

    Returns:
        List of formatted feature terms like "+log(src_port)" or "-sqrt(length)"
    """
    feature_terms = []
    for i, weight in enumerate(weights):
        feature_idx = abs(weight)
        sign = "+" if weight >= 0 else "-"
        feature_name = features[feature_idx] if feature_idx < len(features) else f"Unknown_{feature_idx}"
        transform = transforms[i]
        if transform == 'none':
            feature_terms.append(f"{sign}{feature_name}")
        else:
            feature_terms.append(f"{sign}{transform}({feature_name})")
    return feature_terms


def export_bin_data_to_json(binner, xwt: np.ndarray, ywt: np.ndarray,
                            x_transforms: List[str], y_transforms: List[str],
                            features: List[str], current_class_mapped: int,
                            class_map: Dict[int, int], id_to_name: Dict[str, str],
                            pure_count: float, n_bins: int, save_path: str):
    """Export bin data to JSON format for interactive visualization.

    Creates a JSON file containing bin-level aggregated data including:
    - Bin centers and counts
    - Dominant class per bin
    - Target class purity
    - Class distribution percentages
    - Projection metadata (weights, transforms, features)

    Args:
        binner: Binner object containing binned projection data
        xwt: X-axis projection weights (signed feature indices)
        ywt: Y-axis projection weights (signed feature indices)
        x_transforms: Transformation types for X-axis features
        y_transforms: Transformation types for Y-axis features
        features: List of feature names
        current_class_mapped: Mapped ID of target class
        class_map: Mapping from original class IDs to mapped IDs
        id_to_name: Mapping from class ID (as string) to class name
        pure_count: Number of points in pure bins
        n_bins: Number of bins per dimension
        save_path: Output JSON file path

    Side Effects:
        Creates directories as needed and writes JSON file to save_path
    """
    bins = binner.get_bins()

    # Prepare data arrays
    bin_center_x = []
    bin_center_y = []
    bin_counts = []
    dominant_class_id = []
    dominant_class_name = []
    target_class_purity = []
    class_percentages = []

    # Process each bin
    for i in range(n_bins):
        for j in range(n_bins):
            bin_obj = bins[i][j]
            if bin_obj.count > 0:
                bin_center_x.append(float((i + 0.5) / n_bins))
                bin_center_y.append(float((j + 0.5) / n_bins))
                bin_counts.append(int(bin_obj.count))

                # Find dominant class
                class_counts = bin_obj.class_counts
                dominant_idx = int(np.argmax(class_counts))
                dominant_class_id.append(int(dominant_idx))

                # Get dominant class name
                original_id = [k for k, v in class_map.items() if v == dominant_idx][0]
                class_name = id_to_name.get(str(original_id), f"Unknown ({original_id})")
                dominant_class_name.append(class_name)

                # Calculate target class purity
                target_count = class_counts[current_class_mapped]
                purity = target_count / bin_obj.count if bin_obj.count > 0 else 0.0
                target_class_purity.append(float(purity))

                # Calculate percentage for each class in this bin
                bin_class_percentages = {}
                for mapped_class_id in range(len(class_counts)):
                    count = class_counts[mapped_class_id]
                    percentage = count / bin_obj.count if bin_obj.count > 0 else 0.0

                    # Get original class ID and name
                    original_id = [k for k, v in class_map.items() if v == mapped_class_id][0]
                    class_name_full = id_to_name.get(str(original_id), f"Unknown ({original_id})")

                    if percentage > 0:
                        bin_class_percentages[class_name_full] = {
                            "percentage": float(percentage),
                            "count": int(count),
                            "class_id": int(mapped_class_id),
                            "original_class_id": int(original_id)
                        }

                class_percentages.append(bin_class_percentages)

    # Get target class info
    target_original_id = [k for k, v in class_map.items() if v == current_class_mapped][0]
    target_class_name = id_to_name.get(str(target_original_id), f"Unknown ({target_original_id})")

    # Build JSON data structure
    json_data = {
        "bin_center_x": bin_center_x,
        "bin_center_y": bin_center_y,
        "bin_counts": bin_counts,
        "dominant_class_id": dominant_class_id,
        "dominant_class_name": dominant_class_name,
        "target_class_purity": target_class_purity,
        "class_percentages": class_percentages,
        "target_class_id": int(current_class_mapped),
        "target_class_name": target_class_name,
        "axis1_features": format_projection_features(xwt, x_transforms, features),
        "axis2_features": format_projection_features(ywt, y_transforms, features),
        "best_purity_P": float(pure_count),
        "n_bins": int(n_bins),
        "projection_weights_x": xwt.tolist(),
        "projection_weights_y": ywt.tolist(),
        "x_transforms": x_transforms,
        "y_transforms": y_transforms
    }

    # Save JSON file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Bin data exported to: {save_path}")


def export_pair_points_json(x_proj: np.ndarray, y_proj: np.ndarray,
                            class_values_mapped: np.ndarray,
                            class_map: Dict[int, int],
                            id_to_name: Dict[str, str],
                            save_path: str,
                            exclude_normal: bool = False):
    """Export per-point 2D normalized coordinates and labels for a projection pair.

    Creates a JSON file containing individual data points with:
    - PC1, PC2: Normalized projection coordinates in [0,1]
    - label: Class name for each point

    Args:
        x_proj: X-axis projection values (normalized to [0,1])
        y_proj: Y-axis projection values (normalized to [0,1])
        class_values_mapped: Mapped class IDs for each point
        class_map: Mapping from original class IDs to mapped IDs
        id_to_name: Mapping from class ID (as string) to class name
        save_path: Output JSON file path
        exclude_normal: If True, exclude points labeled 'normal' (case-insensitive)

    Side Effects:
        Creates directories as needed and writes JSON file to save_path
    """
    # Invert class map to go from mapped ID back to original ID
    inv_map = {mapped: orig for orig, mapped in class_map.items()}
    points = []
    n = len(x_proj)

    for i in range(n):
        mapped_id = int(class_values_mapped[i])
        orig_id = inv_map.get(mapped_id, None)
        label = id_to_name.get(str(orig_id), f"Unknown ({orig_id})") if orig_id is not None else f"Unknown ({mapped_id})"

        # Skip normal class if requested
        if exclude_normal and isinstance(label, str) and label.strip().lower() == 'normal':
            continue

        points.append({
            'PC1': float(x_proj[i]),
            'PC2': float(y_proj[i]),
            'label': label
        })

    # Save JSON file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({'points': points}, f)

    print(f"Per-point 2D data exported to: {save_path}")
