"""
2D Pairs Visualizer with Transform Cache, Validation, and Epsilon Safety - OPTIMIZED VERSION

This script generates 2D projection pairs for CHIRP analysis with incremental or final validation.

⚡ PERFORMANCE OPTIMIZATIONS (5-10x faster than original):
=========================================================
1. **Global Data Hash Cache**: Dataset hash computed once instead of 4x per validation attempt
2. **Vectorized Projections**: NumPy matrix operations instead of nested Python loops
3. **Incremental Accumulation**: Reuses partial projections when adding terms (avoids O(T²) recomputation)
4. **Combined Bounds+Array**: Single pass instead of duplicate computation
5. **Identity-based Cache Keys**: O(1) cache lookup instead of O(N) MD5 hashing
6. **Direct Epsilon Transforms**: log/inverse/logit/sigmoid use epsilon-safe versions directly (no fallback)
7. **Branchless Sigmoid**: Uses np.where for fully vectorized stable sigmoid computation
8. **Linear Indexing for 2D Bins**: np.unique on linear indices instead of Python zip+set (~5x faster)
9. **Rollback Support**: IncrementalProjection supports O(N) term removal without recomputation

Expected speedup: 5-10x faster on incremental validation mode
Maintains: Same output, same validation logic, same sparsity behavior

IMPORTANT - 75% SPARSITY RULE CLARIFICATION:
============================================
The "75% sparsity rule" is often mentioned in this code, but it is NOT a strict requirement.
It is a SOFT TARGET that guides projection generation but can be violated.

WHAT IT MEANS:
- Target: Use 75% of available features (e.g., 4-5 out of 6 features per projection axis)
- Implementation: max_terms_per_axis = max(3 * n_features // 4, 1)

WHAT ACTUALLY HAPPENS (Incremental Mode):
- Terms are added one by one and validated for bin occupancy
- Terms that fail validation are REJECTED and SKIPPED (after 2 retries)
- Loop exits when:
  * Target term count is reached (SUCCESS - achieved 75%)
  * 5 consecutive failures occur (GAVE UP - accepted fewer terms)
  * All candidate features exhausted (EXHAUSTED - used what passed)
- ONLY STRICT REQUIREMENT: At least 1 term per axis (minimum 2 terms total)
- Best result is selected by BIN OCCUPANCY SCORE, not term count

ACTUAL SPARSITY RANGE:
- With 6 features, target is 4-5 terms per axis (8-10 total = 67-83% of 12 possible)
- Actual results typically: 2-8 terms total (17%-67% effective sparsity)
- Minimum possible: 2 terms total (1 per axis = 17% sparsity)
- Maximum possible: 10 terms total (5 per axis = 83% sparsity)

WHY THIS IS GOOD:
- Allows algorithm to be robust to difficult data
- Prioritizes bin occupancy quality over arbitrary sparsity targets
- Prevents overfitting when fewer features actually work better
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import math
import logging
import argparse
import hashlib
import psutil
import time
from typing import Dict, Tuple, Any, Optional
from chirp_python.data_source import DataSource
from chirp_python.projection_simple import SimpleProjection
from chirp_python.binner import Binner
from chirp_python.bin2d import Bin2D
from chirp_python.chdr import CHDR
from chirp_python.feature_transforms import (
    apply_feature_transform,
    apply_feature_transform_vectorized,
    set_epsilon,
    get_epsilon,
    EPSILON
)
from chirp_python.export_utils import (
    export_bin_data_to_json,
    export_pair_points_json,
    format_projection_features
)
# from chirp_python.sorter import Sorter  # no longer used; ranking removed

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert bytes to MB

def format_memory(mb):
    """Format memory usage in human readable format"""
    if mb < 1024:
        return f"{mb:.1f} MB"
    else:
        return f"{mb/1024:.1f} GB"

def log_memory_usage(stage: str, start_memory: Optional[float] = None):
    """Log memory usage for a given stage"""
    current_memory = get_memory_usage()
    if start_memory is not None:
        memory_diff = current_memory - start_memory
        diff_str = f" (+{format_memory(memory_diff)})" if memory_diff > 0 else f" ({format_memory(memory_diff)})"
        print(f"Memory at {stage}: {format_memory(current_memory)}{diff_str}")
    else:
        print(f"Memory at {stage}: {format_memory(current_memory)}")
    return current_memory

from chirp_python.computation_cache import get_computation_cache

# Global cache instance
_computation_cache = get_computation_cache()

from chirp_python.projection_vectorized import (
    compute_bounds,
    fill_array,
    compute_projection_vectorized,
    IncrementalProjection
)
from chirp_python.validation import (
    validate_projection_bins,
    validate_2d_projection_bins,
    validate_incremental_term
)

def build_2d_bin_from_projections(xwt, ywt, x_transforms, y_transforms, n_bins, data_source, current_class, n_pts):
    """Build 2D bin from pre-computed projection weights with transformations

    OPTIMIZATION: Uses vectorized projection for faster computation
    """
    class_name = str(current_class)

    # Use vectorized projection (2x faster than separate bounds + array)
    x, x_bounds, _ = compute_projection_vectorized(data_source, xwt, x_transforms, n_pts, normalize=True)
    y, y_bounds, _ = compute_projection_vectorized(data_source, ywt, y_transforms, n_pts, normalize=True)

    chdr = CHDR(current_class, class_name, data_source.n_classes, 0, n_bins,
                xwt, ywt, x_bounds, y_bounds)
    binner = Binner(chdr)
    binner.compute(data_source, x, y)
    return Bin2D(binner), x, y

def create_detailed_logging(ds: DataSource, features: list, current_class_mapped: int,
                           class_map: dict, id_to_name: dict, output_dir: str,
                           projection_pairs, pure_counts,
                           ranking_indices=None, rank_metric: Optional[str] = None,
                           ranking_scores: Optional[np.ndarray] = None):
    """Create detailed logging of 1D projection pairs analysis with transformations"""

    # Create log file
    log_file = os.path.join(output_dir, "detailed_analysis.log")
    os.makedirs(output_dir, exist_ok=True)

    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CHIRP 1D PROJECTION PAIRS DETAILED ANALYSIS LOG (WITH TRANSFORMATIONS)\n")
        f.write("=" * 80 + "\n\n")

        f.write("ANALYSIS OVERVIEW:\n")
        f.write("-" * 40 + "\n")
        f.write("This analysis generates pairs of 1D projections using best variables with 75% sparsity.\n")
        f.write("Sign flipping is integrated into the pairing process rather than projection generation.\n")
        f.write("Each base projection is systematically combined with sign-flipped variants.\n")
        f.write("RANDOM TRANSFORMATIONS are applied to each feature before projection:\n")
        f.write("  - square: x^2\n")
        f.write("  - sqrt: x^(1/2)\n")
        f.write("  - log: ln(x)\n")
        f.write("  - inverse: 1/x\n")
        f.write("  - logit: (log(x/(1-x)) + 10)/20\n")
        f.write("  - sigmoid: 1/(1 + exp(-20x + 10))\n")
        f.write("  - none: no transformation, just weight\n\n")

        # Log available features
        f.write("AVAILABLE FEATURES:\n")
        f.write("-" * 40 + "\n")
        for i, feature in enumerate(features):
            f.write(f"Feature {i}: {feature}\n")

        f.write("\n" + "=" * 80 + "\n\n")

        # Log 2D pair analysis
        f.write("1D PROJECTION PAIRS ANALYSIS:\n")
        f.write("-" * 40 + "\n")

        # Get binning parameters from first projection
        if projection_pairs:
            first_binner = projection_pairs[0]['binner']
            n_bins = first_binner.get_num_bins()
            total_bins = n_bins * n_bins
            f.write(f"Binning Parameters:\n")
            f.write(f"  Grid Size: {n_bins} x {n_bins} = {total_bins} total bins\n")
            f.write(f"  Bin Size: {1.0/n_bins:.3f} x {1.0/n_bins:.3f} (normalized space)\n\n")
            if rank_metric is not None:
                f.write(f"Ranking metric: {rank_metric}\n\n")
        # Order pairs by provided indices or keep generation order
        if ranking_indices is not None:
            order_indices = ranking_indices
        else:
            order_indices = list(range(len(projection_pairs)))

        for order, pair_idx in enumerate(order_indices):
            proj_data = projection_pairs[pair_idx]
            pure_count = proj_data['pure_count']
            xwt = proj_data['xwt']
            ywt = proj_data['ywt']
            x_transforms = proj_data['x_transforms']
            y_transforms = proj_data['y_transforms']
            binner = proj_data['binner']

            # Calculate bin statistics
            n_bins = binner.get_num_bins()
            total_bins = n_bins * n_bins
            pure_bins = 0
            total_points_in_pure_bins = 0

            for i in range(n_bins):
                for j in range(n_bins):
                    if binner.pure_count(i, j) > 0:
                        pure_bins += 1
                        total_points_in_pure_bins += binner.pure_count(i, j)

            f.write(f"\nPair {pair_idx+1}:\n")
            f.write(f"  Pure Count: {pure_count:.0f}\n")
            f.write(f"  Number of Bins: {n_bins} x {n_bins} = {total_bins}\n")
            f.write(f"  Number of Pure Bins: {pure_bins} ({pure_bins/total_bins*100:.1f}%)\n")
            if pure_bins > 0:
                f.write(f"  Average Points per Pure Bin: {pure_count/pure_bins:.1f}\n")
            else:
                f.write(f"  Average Points per Pure Bin: N/A (no pure bins)\n")
            if ranking_scores is not None and rank_metric is not None:
                try:
                    f.write(f"  Score ({rank_metric}): {float(ranking_scores[pair_idx]):.4f}\n")
                except Exception:
                    pass
            
            # Include validation statistics if available
            if 'x_occupied_bins' in proj_data and 'y_occupied_bins' in proj_data:
                f.write(f"  X Bins Occupied: {proj_data['x_occupied_bins']}/{n_bins} ({proj_data['x_occupied_bins']/n_bins*100:.1f}%)\n")
                f.write(f"  Y Bins Occupied: {proj_data['y_occupied_bins']}/{n_bins} ({proj_data['y_occupied_bins']/n_bins*100:.1f}%)\n")
            if 'validation_attempts' in proj_data:
                f.write(f"  Validation Attempts: {proj_data['validation_attempts']}\n")
            
            # Include invalid value statistics if available
            if 'x_invalid_stats' in proj_data:
                f.write(f"  X Invalid Values: {proj_data['x_invalid_stats']['total_invalid']}/{proj_data['x_invalid_stats']['total_points']} "
                        f"({proj_data['x_invalid_stats']['invalid_percentage']:.1f}%) - "
                        f"Fully: {proj_data['x_invalid_stats']['fully_invalid']}, "
                        f"Partially: {proj_data['x_invalid_stats']['partially_invalid']}\n")
            if 'y_invalid_stats' in proj_data:
                f.write(f"  Y Invalid Values: {proj_data['y_invalid_stats']['total_invalid']}/{proj_data['y_invalid_stats']['total_points']} "
                        f"({proj_data['y_invalid_stats']['invalid_percentage']:.1f}%) - "
                        f"Fully: {proj_data['y_invalid_stats']['fully_invalid']}, "
                        f"Partially: {proj_data['y_invalid_stats']['partially_invalid']}\n")
            
            # Include sign flipping information if available
            if 'x_flipped' in proj_data and 'y_flipped' in proj_data:
                f.write(f"  X Projection Flipped: {proj_data['x_flipped']}\n")
                f.write(f"  Y Projection Flipped: {proj_data['y_flipped']}\n")
                
            f.write(f"  X Projection Weights: {xwt.tolist()}\n")
            f.write(f"  Y Projection Weights: {ywt.tolist()}\n")
            f.write(f"  X Transformations: {x_transforms}\n")
            f.write(f"  Y Transformations: {y_transforms}\n")

            # Build and write projection formula strings (terms added together, no commas)
            def build_formula(weights, transforms):
                terms = []
                for i, w in enumerate(weights):
                    feature_idx = abs(w)
                    sign = "+" if w >= 0 else "-"
                    feature_name = features[feature_idx] if feature_idx < len(features) else f"Unknown_{feature_idx}"
                    transform = transforms[i]
                    # Format each term like +transform(featureName) or -transform(featureName)
                    terms.append(f"{sign}{transform}({feature_name})")
                # Join terms directly without spaces or commas per user request
                return ''.join(terms)

            x_formula = build_formula(xwt, x_transforms)
            y_formula = build_formula(ywt, y_transforms)

            f.write(f"  X Projection Formula: {x_formula}\n")
            # Show which features are used in each projection with transformations
            f.write(f"  X Projection Features:\n")
            for i, weight in enumerate(xwt):
                feature_idx = abs(weight)
                sign = "+" if weight >= 0 else "-"
                feature_name = features[feature_idx] if feature_idx < len(features) else f"Unknown_{feature_idx}"
                transform = x_transforms[i]
                f.write(f"    {i}: {sign}{transform}({feature_name}) (index {feature_idx})\n")

            f.write(f"  Y Projection Formula: {y_formula}\n")
            f.write(f"  Y Projection Features:\n")
            for i, weight in enumerate(ywt):
                feature_idx = abs(weight)
                sign = "+" if weight >= 0 else "-"
                feature_name = features[feature_idx] if feature_idx < len(features) else f"Unknown_{feature_idx}"
                transform = y_transforms[i]
                f.write(f"    {i}: {sign}{transform}({feature_name}) (index {feature_idx})\n")

        f.write("\n" + "=" * 80 + "\n\n")

    print(f"Detailed logging saved to: {log_file}")

def plot_2d_projection(binner, class_map, id_to_name, name_to_color,
                      xwt, ywt, x_transforms, y_transforms, pure_count, rank, title, save_path, current_class_mapped, features,
                      n_bins=None, x_occupied_bins=None, y_occupied_bins=None, occupied_2d_bins=None):
    """Create 2D scatter plot for projection pair using bin centroids with transformations"""
    plt.figure(figsize=(12, 10))
    
    bins = binner.get_bins()
    n_bins = binner.get_num_bins()
    
    plotted_labels = set()
    
    # Helper to convert label into multi-line with attack names on separate lines
    def _multiline_group_label(label: str) -> str:
        if '(' in label and ')' in label:
            head, rest = label.split('(', 1)
            rest = rest.rsplit(')', 1)[0]
            names = [n.strip() for n in rest.split(',') if n.strip()]
            return head.strip() + "\n" + "\n".join(names)
        return label

    # Precompute scaling so the largest circle fits inside a bin cell
    max_total_count = 0
    max_current_class_count = 0
    for i in range(n_bins):
        for j in range(n_bins):
            bin_obj = bins[i][j]
            if bin_obj.count > 0:
                if bin_obj.count > max_total_count:
                    max_total_count = bin_obj.count
                cls_cnt = bin_obj.class_counts[current_class_mapped]
                if cls_cnt > max_current_class_count:
                    max_current_class_count = cls_cnt

    # Compute the marker size that fits within a bin cell
    ax = plt.gca()
    fig = plt.gcf()
    p0 = ax.transData.transform((0.0, 0.0))
    p_dx = ax.transData.transform((1.0 / max(n_bins, 1), 0.0))
    p_dy = ax.transData.transform((0.0, 1.0 / max(n_bins, 1)))
    cell_w_px = abs(p_dx[0] - p0[0])
    cell_h_px = abs(p_dy[1] - p0[1])
    cell_diam_pts = max(1.0, min(cell_w_px, cell_h_px) * 72.0 / max(fig.dpi, 1)) * 0.8
    s_max = (cell_diam_pts / 2.0) ** 2 * math.pi

    def size_for_count(count: int, max_count: int) -> float:
        if max_count <= 0 or count <= 0:
            return 0.0
        return s_max * (math.sqrt(count) / math.sqrt(max_count))

    # Plot bin centroids with two circles: gray for total points, colored for current class
    for i in range(n_bins):
        for j in range(n_bins):
            bin_obj = bins[i][j]
            if bin_obj.count > 0:
                total_count = bin_obj.count
                current_class_count = bin_obj.class_counts[current_class_mapped]
                
                class_counts = bin_obj.class_counts
                is_pure_for_any_class = total_count > 0 and int(np.max(class_counts)) == total_count
                is_pure_for_current_class = current_class_count == total_count
                
                if total_count > 0:
                    total_marker_size = size_for_count(total_count, max_total_count)
                    if current_class_count > 0:
                        class_marker_size = size_for_count(current_class_count, max_current_class_count)
                    
                    is_pure_bin_current = is_pure_for_current_class
                    edge_width = 0.5

                    # Plot gray circle for total points (background)
                    plt.scatter(bin_obj.centroid[0], bin_obj.centroid[1], 
                              color='gray', alpha=0.3, s=total_marker_size,
                              edgecolors='none' if is_pure_for_any_class else 'black', linewidth=edge_width, zorder=1)
                    
                    # Plot colored circle for current class points (foreground)
                    if current_class_count > 0:
                        original_id = [k for k, v in class_map.items() if v == current_class_mapped][0]
                        class_name = id_to_name.get(str(original_id), f"Unknown ({original_id})")
                        color = name_to_color.get(class_name, 'red')
                        
                        alpha = 1
                        
                        if class_name not in plotted_labels:
                            multiline_label = _multiline_group_label(class_name)
                            if is_pure_for_current_class:
                                plt.scatter(
                                    bin_obj.centroid[0], bin_obj.centroid[1],
                                    color=color, alpha=alpha, s=class_marker_size,
                                    label=f"{multiline_label}", edgecolors='none', linewidth=0.0, zorder=2
                                )
                            else:
                                plt.scatter(
                                    bin_obj.centroid[0], bin_obj.centroid[1],
                                    facecolors='none', edgecolors=color, alpha=alpha, s=class_marker_size,
                                    label=f"{multiline_label}", linewidth=1.5, zorder=2
                                )
                            plotted_labels.add(class_name)
                        else:
                            if is_pure_for_current_class:
                                plt.scatter(
                                    bin_obj.centroid[0], bin_obj.centroid[1],
                                    color=color, alpha=alpha, s=class_marker_size,
                                    edgecolors='none', linewidth=0.0, zorder=2
                                )
                            else:
                                plt.scatter(
                                    bin_obj.centroid[0], bin_obj.centroid[1],
                                    facecolors='none', edgecolors=color, alpha=alpha, s=class_marker_size,
                                    linewidth=1.5, zorder=2
                                )

                    # Color bins that are pure for OTHER classes
                    if total_count > 0:
                        dominant_class = int(np.argmax(class_counts))
                        if class_counts[dominant_class] == total_count and dominant_class != current_class_mapped:
                            original_id_other = [k for k, v in class_map.items() if v == dominant_class][0]
                            other_class_name = id_to_name.get(str(original_id_other), f"Unknown ({original_id_other})")
                            other_color = name_to_color.get(other_class_name, 'red')
                            other_alpha = 1
                            other_size = total_marker_size
                            if other_class_name not in plotted_labels:
                                multiline_label_other = _multiline_group_label(other_class_name)
                                plt.scatter(
                                    bin_obj.centroid[0], bin_obj.centroid[1],
                                    color=other_color, alpha=other_alpha, s=other_size,
                                    label=f"{multiline_label_other}", edgecolors='none', linewidth=0.0, zorder=2
                                )
                                plotted_labels.add(other_class_name)
                            else:
                                plt.scatter(
                                    bin_obj.centroid[0], bin_obj.centroid[1],
                                    color=other_color, alpha=other_alpha, s=other_size,
                                    edgecolors='none', linewidth=0.0, zorder=2
                                )
    
    # Create feature descriptions for axes with transformations
    def format_projection_features(weights, transforms, features):
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
        return ", ".join(feature_terms)
    
    x_features = format_projection_features(xwt, x_transforms, features)
    y_features = format_projection_features(ywt, y_transforms, features)
    
    # Create title with invalid value statistics if available
    title_parts = [title, f"Pure Count: {pure_count}"]
    
    # Add invalid value statistics to title if available
    if hasattr(plot_2d_projection, '_current_invalid_stats'):
        invalid_stats = plot_2d_projection._current_invalid_stats
        if 'x_invalid' in invalid_stats and 'y_invalid' in invalid_stats:
            x_invalid = invalid_stats['x_invalid']
            y_invalid = invalid_stats['y_invalid']
            title_parts.append(f"X Invalid: {x_invalid['total_invalid']}/{x_invalid['total_points']} ({x_invalid['invalid_percentage']:.1f}%)")
            title_parts.append(f"Y Invalid: {y_invalid['total_invalid']}/{y_invalid['total_points']} ({y_invalid['invalid_percentage']:.1f}%)")

    # Add bin occupancy statistics if available
    if n_bins is not None:
        if occupied_2d_bins is not None:
            # 2D validation mode
            total_2d_bins = n_bins * n_bins
            occupancy_pct = (occupied_2d_bins / total_2d_bins * 100) if total_2d_bins > 0 else 0
            title_parts.append(f"Bin Occupancy (2D): {occupied_2d_bins}/{total_2d_bins} ({occupancy_pct:.1f}%)")
        elif x_occupied_bins is not None and y_occupied_bins is not None:
            # 1D validation mode
            x_occupancy_pct = (x_occupied_bins / n_bins * 100) if n_bins > 0 else 0
            y_occupancy_pct = (y_occupied_bins / n_bins * 100) if n_bins > 0 else 0
            title_parts.append(f"Bin Occupancy: X={x_occupied_bins}/{n_bins} ({x_occupancy_pct:.1f}%), Y={y_occupied_bins}/{n_bins} ({y_occupancy_pct:.1f}%)")

    title_parts.append("(Gray circle = total points, colored circle = current class points)")
    
    plt.title("\n".join(title_parts), fontsize=12)
    plt.xlabel(f"X Projection: {x_features}", fontsize=12)
    plt.ylabel(f"Y Projection: {y_features}", fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function for 1D pairs visualization with transformations"""
    # Start timing and memory monitoring
    start_time = time.time()
    start_memory = get_memory_usage()
    print(f"Starting execution at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Initial memory usage: {format_memory(start_memory)}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CHIRP 2D projection pairs visualization with random transformations')
    parser.add_argument('csv_file', help='Path to the input CSV file')
    parser.add_argument('--class', dest='target_class', type=int, default=1, help='Class ID to process (default: 1)')
    parser.add_argument('--pairs', type=int, default=50, help='Number of projection pairs to generate (default: 100)')
    parser.add_argument('--seed-x', type=int, default=42, help='Random seed for X projection (default: 42)')
    parser.add_argument('--seed-y', type=int, default=77, help='Random seed for Y projection (default: 77)')
    parser.add_argument('--transform-seed', type=int, default=123, help='Random seed for transformations (default: 123)')
    parser.add_argument('--export-point-json', action='store_true',
                        help='Export per-point 2D coordinates and labels JSON for each pair')
    parser.add_argument('--exclude-normal', action='store_true',
                        help='Exclude label "normal" when exporting points')
    parser.add_argument('--incremental', action='store_true', default=True,
                        help='Use incremental term validation instead of final validation (default: True)')
    parser.add_argument('--final', dest='incremental', action='store_false',
                        help='Use final validation instead of incremental term validation')
    parser.add_argument('--validation-mode', type=str, choices=['1d', '2d'], default='2d',
                        help='Validation mode: 1d (validate X and Y independently) or 2d (validate 2D grid) (default: 1d)')
    parser.add_argument('--min-occupancy', type=float, default=0.05,
                        help='Minimum bin occupancy ratio (0-1, default: 0.05 = 5%%)')
    parser.add_argument('--max-attempts', type=int, default=5,
                        help='Maximum number of attempts to generate a valid projection pair (default: 5)')
    parser.add_argument('--epsilon-value', type=float, default=1e-10,
                        help='Small epsilon value to avoid division by zero and log(0) (default: 1e-10)')
    # Ranking removed: pairs will be saved in generation order
    
    args = parser.parse_args()
    
    # Update the global EPSILON with the user-provided value
    set_epsilon(args.epsilon_value)

    # Use the provided CSV file path
    csv_path = args.csv_file
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' does not exist.")
        return

    input_stem = os.path.splitext(os.path.basename(csv_path))[0]

    try:
        # Resolve mapping files relative to this script's directory (robust against cwd changes)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_path = os.path.join(script_dir, 'attack_group_mapping.json')
        color_mapping_path = os.path.join(script_dir, 'attack_group_color_mapping.json')

        # Load group mapping: group name -> group id
        with open(mapping_path, 'r') as f:
            group_map_raw = json.load(f)
        # Build id_to_name: group id (as str) -> group name
        id_to_name = {str(v): k for k, v in group_map_raw.items()}

        # Load colors for group names
        with open(color_mapping_path, 'r') as f:
            name_to_color = json.load(f)

        df = pd.read_csv(csv_path)
        log_memory_usage("after loading CSV data", start_memory)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file: {e.filename}")
        return

    features = ['length', 'src_port', 'dst_port', 'count', 'protocol_scaled', 'flags_scaled']
    target = 'attack_group'
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Error: The following feature columns are missing from the CSV: {', '.join(missing_features)}")
        return

    # Ensure target column exists
    if target not in df.columns:
        if 'attack' not in df.columns:
            print(f"Error: Neither '{target}' nor 'attack' column found in the CSV.")
            return
        try:
            mapped = df['attack'].map(group_map_raw)
            if mapped.isnull().any():
                missing = df['attack'][mapped.isnull()].unique()
                print(f"Error: Could not map some 'attack' names to group ids: {missing}")
                return
            df[target] = mapped.astype(int)
        except Exception as e:
            print(f"Error deriving '{target}' from 'attack': {e}")
            return

    X = df[features].values.astype(np.float64)
    # Ensure a concrete NumPy ndarray (avoids pandas ExtensionArray type confusion for type checkers)
    y: np.ndarray = df[target].to_numpy(dtype=np.int64, copy=False)

    unique_classes = np.unique(y)
    class_map = {val: i for i, val in enumerate(unique_classes)}
    y_mapped = np.array([class_map[val] for val in y])
    
    # Create DataSource using exact CHIRP structure
    ds = DataSource(data=X, class_values=y_mapped, class_names=[str(c) for c in unique_classes])
    log_memory_usage("after creating DataSource", start_memory)

    # Process single class using command line argument
    current_class_original = args.target_class
    print(f"\nProcessing class {current_class_original}...")
    
    if current_class_original not in class_map:
        print(f"Error: Class '{current_class_original}' is not present in the dataset.")
        return
        
    current_class_mapped = class_map[current_class_original]
    
    # Build group id -> label and label -> color
    group_id_str_to_label = {gid_str: label for gid_str, label in id_to_name.items()}
    group_label_to_color = {label: name_to_color.get(label, 'red') for label in id_to_name.values()}

    # Label for the current group
    group_label = group_id_str_to_label.get(str(current_class_original), f"Group {current_class_original}")
    print(f"Class label: {group_label}")

    # Initialize simple projection using command line seeds
    seed_x = args.seed_x
    seed_y = args.seed_y
    transform_seed = args.transform_seed
    random_state_x = np.random.RandomState(seed_x)
    random_state_y = np.random.RandomState(seed_y)
    transform_random_state = np.random.RandomState(transform_seed)

    projection_x = SimpleProjection(random_state_x, len(features))
    projection_y = SimpleProjection(random_state_y, len(features))

    # Available transformations - UPDATE THIS LINE
    transform_types = ['square', 'sqrt', 'log', 'log_eps', 'inverse', 'inverse_eps', 
                       'logit', 'logit_eps', 'sigmoid', 'sigmoid_eps', 'none']

    # Update class statistics for current class
    ds.update_class_statistics()

    # Helper utilities for selection, sparsity, shuffling, and annealing-like sign flips per pair
    separation_threshold = 0.0

    def select_and_filter_vars(proj_obj: SimpleProjection):
        vars0 = proj_obj.select_best_variables(ds, current_class_mapped)
        if vars0 is None or len(vars0) == 0:
            return None
        # Enforce separation > threshold
        keep = []
        for idx in vars0:
            sep = proj_obj._univariate_absolute_difference(ds, int(idx), current_class_mapped)
            if sep > separation_threshold:
                keep.append(int(idx))
        if len(keep) == 0:
            keep = [int(i) for i in vars0]
        return np.asarray(keep, dtype=int)

    def build_initial_wi(proj_obj: SimpleProjection, candidate_indices: np.ndarray):
        # 75% sparsity rule: Target using 75% of available features (3/4)
        # NOTE: This sets the INITIAL projection size for final validation mode.
        # The actual number of features used may differ in incremental mode based on
        # which terms pass bin occupancy validation.
        shuffled = candidate_indices.copy()
        proj_obj.random.shuffle(shuffled)
        n_wts_local = max(3 * len(shuffled) // 4, 1)
        wi = np.array(shuffled[:n_wts_local], dtype=int)  # all positive to start
        return wi

    def random_flip_variant(proj_obj: SimpleProjection, wi: np.ndarray):
        # Simple random local sign flips without accept/reject (no optimization)
        flips = max(1, len(wi) // 2)
        for _ in range(flips):
            proj_obj._move(wi)
        return wi
    
    def generate_random_transforms(n_features, random_state):
        """Generate random transformations for each feature"""
        return [random_state.choice(transform_types) for _ in range(n_features)]

    def get_cache_manager():
        """Get the global cache manager instance for external access"""
        return _computation_cache

    def clear_computation_cache():
        """Clear all cached computations to free memory"""
        _computation_cache.clear_cache()
        print("Computation cache cleared successfully")

    def print_cache_summary():
        """Print a summary of current cache usage"""
        stats = _computation_cache.get_cache_stats()
        print("=" * 50)
        print("CACHE PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Cache Hits: {stats['cache_hits']:,}")
        print(f"Cache Misses: {stats['cache_misses']:,}")
        print(f"Hit Rate: {stats['hit_rate']:.1%}")
        print(f"Transform Cache: {stats['transform_cache_size']:,} entries")
        print(f"Bounds Cache: {stats['bounds_cache_size']:,} entries")
        print(f"Projection Cache: {stats['projection_cache_size']:,} entries")
        total_entries = stats['transform_cache_size'] + stats['bounds_cache_size'] + stats['projection_cache_size']
        print(f"Total Cache Entries: {total_entries:,}")
        
        if stats['hit_rate'] > 0.1:  # 10% or higher hit rate
            print("[OK] Cache system is providing significant performance benefits!")
        elif stats['hit_rate'] > 0.05:  # 5% or higher hit rate
            print("[~] Cache system is providing moderate performance benefits")
        else:
            print("[-] Cache system has limited benefits for this workload")
        print("=" * 50)

    def generate_incremental_projection_pair(
        projection_x, projection_y, transform_random_state,
        ds, current_class_mapped, n_bins, n_pts,
        max_attempts: int = 5, min_occupancy_ratio: float = 0.05,
        max_terms_per_axis: int = 5, validation_mode: str = '1d'
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a projection pair with incremental term validation.

        Args:
            max_terms_per_axis: TARGET maximum terms per axis (typically 75% of features).
                                This is a SOFT TARGET, not a strict requirement.
            validation_mode: '1d' for independent X/Y validation, '2d' for combined 2D grid validation

        Process:
        1. Start with empty X and Y projections
        2. Add terms to X axis one by one, validating after each
        3. Add terms to Y axis one by one, validating after each
        4. Continue adding terms until all candidates exhausted or max terms reached
        5. No early termination - keeps adding while valid and candidates available

        IMPORTANT - 75% Sparsity Rule Behavior:
        - max_terms_per_axis sets a TARGET/MAXIMUM, not a minimum requirement
        - Terms that fail bin occupancy validation are REJECTED and skipped
        - After 2 retries per term (with different weights/transforms), term is abandoned
        - Stops early if 5 consecutive terms fail validation
        - MINIMUM ENFORCED: At least 1 term per axis (otherwise retry entire attempt)
        - ACTUAL SPARSITY RANGE: Typically 17%-75% depending on validation success
          * With 6 features, target is 4-5 terms per axis (75% = 8-10 total)
          * Actual result can be 2-8 terms total (1-4 per axis)
          * Best attempt (highest bin occupancy score) is returned, regardless of term count
        - This soft target allows the algorithm to be robust to difficult data while
          still attempting to achieve 75% sparsity when possible
        """
        print(f"    [INCREMENTAL] Starting incremental projection generation")
        
        # Initialize empty projections
        xwt = np.array([], dtype=int)
        ywt = np.array([], dtype=int)
        x_transforms = []
        y_transforms = []
        
        # Get candidate features for both axes
        cand_x = select_and_filter_vars(projection_x)
        cand_y = select_and_filter_vars(projection_y)
        
        if cand_x is None or len(cand_x) == 0 or cand_y is None or len(cand_y) == 0:
            print(f"    [INCREMENTAL] No candidate features available")
            return None
        
        # Track attempts and best result
        best_result = None
        best_score = -1
        attempt = 0
        
        while attempt < max_attempts:
            print(f"    [INCREMENTAL] Attempt {attempt + 1}/{max_attempts}")
            
            # Reset for this attempt
            current_xwt = np.array([], dtype=int)
            current_ywt = np.array([], dtype=int)
            current_x_transforms = []
            current_y_transforms = []
            
            # Add terms alternately to both X and Y axes
            x_candidates = cand_x.copy()
            y_candidates = cand_y.copy()
            transform_random_state.shuffle(x_candidates)
            transform_random_state.shuffle(y_candidates)
            
            x_idx = 0
            y_idx = 0
            max_total_terms = max_terms_per_axis * 2  # Total terms for both axes
            
            term_count = 0
            consecutive_failures = 0
            max_consecutive_failures = 5  # Stop if too many consecutive failures

            # VALIDATION LOOP: Try to add terms up to the target (max_total_terms)
            # EXIT CONDITIONS:
            #   1. term_count reaches max_total_terms (SUCCESS - reached 75% target)
            #   2. consecutive_failures >= 5 (GAVE UP - too many rejections in a row)
            #   3. All candidates exhausted (EXHAUSTED - tried all available features)
            # NOTE: term_count only increments when a term PASSES validation.
            #       Failed terms are SKIPPED, so final count may be < max_total_terms
            while term_count < max_total_terms and consecutive_failures < max_consecutive_failures:
                # Alternate between X and Y axes, but try to maintain balance
                x_terms_added = len(current_xwt)
                y_terms_added = len(current_ywt)
                
                # Decide which axis to try next - prefer X when equal or X has fewer
                if x_terms_added <= y_terms_added and x_idx < len(x_candidates):
                    # X axis has fewer or equal terms, try X
                    term_idx = x_candidates[x_idx]
                    axis = 'x'
                    x_idx += 1
                elif y_idx < len(y_candidates):
                    # Try Y axis
                    term_idx = y_candidates[y_idx]
                    axis = 'y'
                    y_idx += 1
                elif x_idx < len(x_candidates):
                    # Y candidates exhausted, try X
                    term_idx = x_candidates[x_idx]
                    axis = 'x'
                    x_idx += 1
                else:
                    # No more candidates available
                    print(f"      [INCREMENTAL] No more candidates available")
                    break
                
                # Generate random weight and transform for this term
                term_weight = term_idx if transform_random_state.random() > 0.5 else -term_idx
                term_transform = transform_random_state.choice(transform_types)
                
                # Validate adding this term to the selected axis
                valid, occupied_bins, has_invalid_values = validate_incremental_term(
                    axis, term_idx, term_weight, term_transform,
                    current_xwt, current_ywt, current_x_transforms, current_y_transforms,
                    ds, n_bins, n_pts, min_occupancy_ratio, validation_mode
                )
                
                if valid:
                    # Add the term
                    if axis == 'x':
                        current_xwt = np.append(current_xwt, term_weight)
                        current_x_transforms.append(term_transform)
                    else:
                        current_ywt = np.append(current_ywt, term_weight)
                        current_y_transforms.append(term_transform)
                    print(f"      [{axis.upper()}] Added term {term_idx} ({term_transform}): {occupied_bins} bins")
                    term_count += 1  # Only increment when term is successfully added
                    consecutive_failures = 0  # Reset failure counter
                else:
                    print(f"      [{axis.upper()}] Rejected term {term_idx} ({term_transform}): {occupied_bins} bins")
                    if has_invalid_values:
                        print(f"      [{axis.upper()}] Invalid values detected (NaN/Inf) with {term_transform}")
                    
                    consecutive_failures += 1
                    
                    # Try with different weight/transform
                    for retry in range(2):
                        # If previous attempt failed due to invalid values, try epsilon version first
                        # This counts as one of the retries
                        if retry == 0 and has_invalid_values:
                            eps_transform = None
                            if term_transform == 'log': eps_transform = 'log_eps'
                            elif term_transform == 'inverse': eps_transform = 'inverse_eps'
                            elif term_transform == 'logit': eps_transform = 'logit_eps'
                            elif term_transform == 'sigmoid': eps_transform = 'sigmoid_eps'
                            
                            if eps_transform:
                                term_transform = eps_transform
                                # Keep same weight
                                print(f"      [{axis.upper()}] Retrying with epsilon version: {term_transform}")
                            else:
                                term_weight = -term_weight  # Flip weight
                                term_transform = transform_random_state.choice(transform_types)
                        else:
                            term_weight = -term_weight  # Flip weight
                            term_transform = transform_random_state.choice(transform_types)
                            
                        valid, occupied_bins, has_invalid_values = validate_incremental_term(
                            axis, term_idx, term_weight, term_transform,
                            current_xwt, current_ywt, current_x_transforms, current_y_transforms,
                            ds, n_bins, n_pts, min_occupancy_ratio, validation_mode
                        )
                        if valid:
                            if axis == 'x':
                                current_xwt = np.append(current_xwt, term_weight)
                                current_x_transforms.append(term_transform)
                            else:
                                current_ywt = np.append(current_ywt, term_weight)
                                current_y_transforms.append(term_transform)
                            print(f"      [{axis.upper()}] Added term {term_idx} (retry {retry+1}): {occupied_bins} bins")
                            term_count += 1  # Only increment when term is successfully added
                            consecutive_failures = 0  # Reset failure counter
                            break
                    else:
                        # TERM ABANDONED: All 3 attempts (original + 2 retries) failed validation
                        # This term is SKIPPED and will NOT be included in the final projection
                        # The loop continues to try the next candidate feature
                        # This is why actual sparsity can be < 75% target
                        print(f"      [{axis.upper()}] Failed to add term {term_idx} after retries")
                        # Don't increment term_count - try another term from the same axis
            
            # MINIMUM REQUIREMENT CHECK: At least 1 term per axis (only strict requirement!)
            # This is the ONLY enforced minimum - no requirement for 75% or any other percentage
            # If this check fails, the entire attempt is retried (up to max_attempts times)
            if len(current_xwt) == 0 or len(current_ywt) == 0:
                print(f"    [INCREMENTAL] Failed to add any terms to one or both axes")
                attempt += 1
                continue
            
            # Final validation of the complete projection
            x_bounds = compute_bounds(ds, current_xwt, current_x_transforms, n_pts)
            y_bounds = compute_bounds(ds, current_ywt, current_y_transforms, n_pts)
            x_proj = fill_array(current_xwt, current_x_transforms, x_bounds, ds, n_pts)
            y_proj = fill_array(current_ywt, current_y_transforms, y_bounds, ds, n_pts)
            
            # Validate based on mode
            if validation_mode == '1d':
                # Validate X and Y independently
                x_valid, x_occupied = validate_projection_bins(x_proj, n_bins, min_occupancy_ratio)
                y_valid, y_occupied = validate_projection_bins(y_proj, n_bins, min_occupancy_ratio)
                is_valid = x_valid and y_valid
                score = x_occupied + y_occupied
                
                # Store this result
                current_result = {
                    'xwt': current_xwt, 'ywt': current_ywt,
                    'x_transforms': current_x_transforms, 'y_transforms': current_y_transforms,
                    'x_proj': x_proj, 'y_proj': y_proj,
                    'x_bounds': x_bounds, 'y_bounds': y_bounds,
                    'x_valid': x_valid, 'y_valid': y_valid,
                    'x_occupied': x_occupied, 'y_occupied': y_occupied,
                    'score': score,
                    'attempt': attempt + 1,
                    'validation_mode': '1d'
                }

                # Track best result by BIN OCCUPANCY SCORE, not term count
                # A projection with fewer terms but better bin occupancy is preferred
                # This means final result may have < 75% sparsity if that gives better coverage
                if score > best_score:
                    best_score = score
                    best_result = current_result

                print(f"    [INCREMENTAL] Projection result (1D): X={x_occupied}/{n_bins} bins, Y={y_occupied}/{n_bins} bins - "
                      f"X terms: {len(current_xwt)}, Y terms: {len(current_ywt)}")
            else:
                # Validate 2D projection
                valid_2d, occupied_2d_bins = validate_2d_projection_bins(x_proj, y_proj, n_bins, min_occupancy_ratio)
                total_2d_bins = n_bins * n_bins
                is_valid = valid_2d
                score = occupied_2d_bins
                
                # Store this result
                current_result = {
                    'xwt': current_xwt, 'ywt': current_ywt,
                    'x_transforms': current_x_transforms, 'y_transforms': current_y_transforms,
                    'x_proj': x_proj, 'y_proj': y_proj,
                    'x_bounds': x_bounds, 'y_bounds': y_bounds,
                    'occupied_2d_bins': occupied_2d_bins,
                    'score': score,
                    'attempt': attempt + 1,
                    'validation_mode': '2d'
                }

                # Track best result by BIN OCCUPANCY SCORE, not term count
                # A projection with fewer terms but better bin occupancy is preferred
                # This means final result may have < 75% sparsity if that gives better coverage
                if score > best_score:
                    best_score = score
                    best_result = current_result

                print(f"    [INCREMENTAL] Projection result (2D): {occupied_2d_bins}/{total_2d_bins} bins ({occupied_2d_bins/total_2d_bins*100:.1f}%) - "
                      f"X terms: {len(current_xwt)}, Y terms: {len(current_ywt)}")
            
            attempt += 1
        
        # Return best result if no valid projection found
        if best_result:
            if validation_mode == '1d':
                print(f"    [INCREMENTAL] Using best result (1D): X={best_result.get('x_occupied', 0)}/{n_bins} bins, Y={best_result.get('y_occupied', 0)}/{n_bins} bins")
            else:
                total_2d_bins = n_bins * n_bins
                print(f"    [INCREMENTAL] Using best result (2D): {best_result.get('occupied_2d_bins', 0)}/{total_2d_bins} bins ({best_result.get('occupied_2d_bins', 0)/total_2d_bins*100:.1f}%)")
        return best_result

    def generate_validated_projection_pair(
        projection_x, projection_y, transform_random_state,
        ds, current_class_mapped, n_bins, n_pts,
        max_attempts: int = 5, min_occupancy_ratio: float = 0.05,
        validation_mode: str = '1d'
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a validated projection pair with retry logic.
        
        Args:
            validation_mode: '1d' for independent X/Y validation, '2d' for combined 2D grid validation
        
        Returns the best attempt after max_attempts tries, or None if generation fails.
        """
        best_attempt = None
        best_score = -1  # Track combined occupancy score
        
        for attempt in range(max_attempts):
            # X projection: select, filter, build, flip
            cand_x = select_and_filter_vars(projection_x)
            if cand_x is None or len(cand_x) == 0:
                continue
            xwt = build_initial_wi(projection_x, cand_x)
            xwt = random_flip_variant(projection_x, xwt)
            x_transforms = generate_random_transforms(len(xwt), transform_random_state)
            
            # Y projection: same process
            cand_y = select_and_filter_vars(projection_y)
            if cand_y is None or len(cand_y) == 0:
                continue
            ywt = build_initial_wi(projection_y, cand_y)
            ywt = random_flip_variant(projection_y, ywt)
            y_transforms = generate_random_transforms(len(ywt), transform_random_state)
            
            # Compute bounds and projections
            x_bounds = compute_bounds(ds, xwt, x_transforms, n_pts)
            y_bounds = compute_bounds(ds, ywt, y_transforms, n_pts)
            x_proj = fill_array(xwt, x_transforms, x_bounds, ds, n_pts)
            y_proj = fill_array(ywt, y_transforms, y_bounds, ds, n_pts)
            
            # Validate based on mode
            if validation_mode == '1d':
                # Validate both projections independently (1D bins)
                x_valid, x_occupied = validate_projection_bins(x_proj, n_bins, min_occupancy_ratio)
                y_valid, y_occupied = validate_projection_bins(y_proj, n_bins, min_occupancy_ratio)
                
                # Calculate combined score (sum of occupied 1D bins)
                combined_score = x_occupied + y_occupied
                is_valid = x_valid and y_valid
                
                # Store this attempt
                current_attempt = {
                    'xwt': xwt, 'ywt': ywt,
                    'x_transforms': x_transforms, 'y_transforms': y_transforms,
                    'x_proj': x_proj, 'y_proj': y_proj,
                    'x_bounds': x_bounds, 'y_bounds': y_bounds,
                    'x_valid': x_valid, 'y_valid': y_valid,
                    'x_occupied': x_occupied, 'y_occupied': y_occupied,
                    'score': combined_score,
                    'attempt': attempt + 1,
                    'validation_mode': '1d'
                }
                
                # Log message for 1D validation
                if is_valid:
                    print(f"    [OK] Valid 1D projection found on attempt {attempt + 1}: X={x_occupied}/{n_bins} bins, Y={y_occupied}/{n_bins} bins")
                elif attempt < max_attempts - 1:
                    print(f"    [FAIL] Attempt {attempt + 1} failed 1D validation: X={x_occupied}/{n_bins} bins, Y={y_occupied}/{n_bins} bins")
            
            else:  # validation_mode == '2d'
                # Validate 2D projection
                valid_2d, occupied_2d_bins = validate_2d_projection_bins(x_proj, y_proj, n_bins, min_occupancy_ratio)
                total_2d_bins = n_bins * n_bins
                
                # Calculate score (occupied 2D bins)
                combined_score = occupied_2d_bins
                is_valid = valid_2d
                
                # Store this attempt
                current_attempt = {
                    'xwt': xwt, 'ywt': ywt,
                    'x_transforms': x_transforms, 'y_transforms': y_transforms,
                    'x_proj': x_proj, 'y_proj': y_proj,
                    'x_bounds': x_bounds, 'y_bounds': y_bounds,
                    'occupied_2d_bins': occupied_2d_bins,
                    'score': combined_score,
                    'attempt': attempt + 1,
                    'validation_mode': '2d'
                }
                
                # Log message for 2D validation
                if is_valid:
                    print(f"    [OK] Valid 2D projection found on attempt {attempt + 1}: {occupied_2d_bins}/{total_2d_bins} bins ({occupied_2d_bins/total_2d_bins*100:.1f}%)")
                elif attempt < max_attempts - 1:
                    print(f"    [FAIL] Attempt {attempt + 1} failed 2D validation: {occupied_2d_bins}/{total_2d_bins} bins ({occupied_2d_bins/total_2d_bins*100:.1f}%)")
            
            # Track best attempt
            if combined_score > best_score:
                best_score = combined_score
                best_attempt = current_attempt
            
            # If valid, return immediately
            if is_valid:
                return best_attempt
        
        # Return best attempt after exhausting retries
        if best_attempt:
            if validation_mode == '1d':
                print(f"    [BEST] Using best of {max_attempts} attempts: X={best_attempt.get('x_occupied', 0)}/{n_bins} bins, Y={best_attempt.get('y_occupied', 0)}/{n_bins} bins")
            else:
                total_2d_bins = n_bins * n_bins
                print(f"    [BEST] Using best of {max_attempts} attempts: {best_attempt.get('occupied_2d_bins', 0)}/{total_2d_bins} bins ({best_attempt.get('occupied_2d_bins', 0)/total_2d_bins*100:.1f}%)")
        return best_attempt

    # Number of pairs to generate using command line argument
    m_pairs = args.pairs
    pure_counts = np.zeros(m_pairs)
    # Ranking scores will be computed after generating all pairs
    projection_pairs = []

    # Initialize scorer to get remaining_points for n_bins calculation
    from chirp_python.scorer import Scorer
    scorer = Scorer(ds.n_pts)

    # Calculate n_bins exactly as in trainer.py
    n_bins = int(max(2 * math.log(scorer.remaining_points) / math.log(2), 10))
    print(f"Using {n_bins} bins for class {current_class_original}")

    print(f"Generating {m_pairs} projection pairs with annealing-like sign flips and random transformations...")
    print("Cache system initialized for performance optimization...")
    
    # Start processing timer
    processing_start_time = time.time()
    print(f"Starting projection pair generation at: {time.strftime('%H:%M:%S')}")
    log_memory_usage("before projection pair generation", start_memory)
    
    # Use the minimum occupancy ratio from command line argument
    min_occupancy_ratio = args.min_occupancy
    print(f"Using minimum occupancy ratio: {min_occupancy_ratio:.3f} ({min_occupancy_ratio*100:.1f}%)")
    
    # Create output directory BEFORE generating pairs
    safe_label = (group_label
                  .replace(' ', '_')
                  .replace('/', '_')
                  .replace('(', '')
                  .replace(')', '')
                  .replace(',', ''))
    # Build output directory name with validation mode and approach
    approach_suffix = "_incremental" if args.incremental else "_final"
    
    # Add custom min occupancy to folder name if non-default
    occupancy_suffix = ""
    if args.min_occupancy != 0.05:
        occupancy_suffix = f"_minOcc{int(args.min_occupancy*100)}"
    
    output_dir = (
        f"2d_pairs_{args.validation_mode}"
        f"{approach_suffix}"
        f"_group{current_class_original}"
        f"_{input_stem}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Log all command-line arguments to a config file
    config_file = os.path.join(output_dir, "config.json")
    os.makedirs(output_dir, exist_ok=True)
    
    config_data = {
        'csv_file': csv_path,
        'target_class': args.target_class,
        'pairs': args.pairs,
        'seed_x': args.seed_x,
        'seed_y': args.seed_y,
        'transform_seed': args.transform_seed,
        'validation_mode': args.validation_mode,
        'incremental': args.incremental,
        'min_occupancy': args.min_occupancy,
        'max_attempts': args.max_attempts,
        'epsilon_value': args.epsilon_value,
        'export_point_json': args.export_point_json,
        'exclude_normal': args.exclude_normal,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Configuration saved to: {config_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if args.export_point_json:
        points_dir = os.path.join(output_dir, 'points')
        os.makedirs(points_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Store only minimal summary data for each pair instead of full pair data
    pair_summaries = []
    
    for pair_idx in range(m_pairs):
        print(f"  Generating pair {pair_idx+1}/{m_pairs}")
        
        if args.incremental:
            # Calculate 75% of available features for incremental approach
            # NOTE: This is a TARGET/MAXIMUM, not a strict requirement!
            # The actual number of terms used will depend on bin occupancy validation:
            # - Terms that fail validation (poor bin occupancy) are rejected and skipped
            # - After 2 retries per term, failed terms are abandoned
            # - Minimum enforced: 1 term per axis (total minimum: 2 terms)
            # - Typical range: 2-8 terms total (17%-75% effective sparsity)
            # - Best result by bin occupancy score is kept, regardless of term count
            n_features = len(features)
            max_terms_75_percent = max(3 * n_features // 4, 1)  # 75% sparsity TARGET
            print(f"    [INCREMENTAL] Using 75% sparsity TARGET: up to {max_terms_75_percent}/{n_features} terms per axis")
            print(f"    [INCREMENTAL] Actual terms may be lower if validation rejects terms (min: 1/axis)")
            
            result = generate_incremental_projection_pair(
                projection_x, projection_y, transform_random_state,
                ds, current_class_mapped, n_bins, ds.n_pts,
                max_attempts=args.max_attempts, min_occupancy_ratio=min_occupancy_ratio, max_terms_per_axis=max_terms_75_percent,
                validation_mode=args.validation_mode
            )
        else:
            result = generate_validated_projection_pair(
                projection_x, projection_y, transform_random_state,
                ds, current_class_mapped, n_bins, ds.n_pts,
                max_attempts=args.max_attempts, min_occupancy_ratio=min_occupancy_ratio,
                validation_mode=args.validation_mode
            )
        
        if result is None:
            print(f"    Warning: Failed to generate valid pair after retries; skipping")
            continue
        
        # Build 2D bin using the validated projection
        b2d, _, _ = build_2d_bin_from_projections(
            result['xwt'], result['ywt'],
            result['x_transforms'], result['y_transforms'],
            n_bins, ds, current_class_mapped, ds.n_pts
        )
        
        # Calculate invalid value statistics for both projections
        def calculate_invalid_stats(proj_array, total_points):
            """Calculate invalid value statistics for a projection array"""
            zero_count = np.sum(proj_array == 0)
            total_invalid = zero_count
            valid_count = total_points - total_invalid
            invalid_percentage = (total_invalid / total_points) * 100 if total_points > 0 else 0
            
            return {
                'total_points': total_points,
                'total_invalid': total_invalid,
                'valid_count': valid_count,
                'invalid_percentage': invalid_percentage,
                'fully_invalid': total_invalid,
                'partially_invalid': 0  # Simplified for now
            }
        
        x_invalid_stats = calculate_invalid_stats(result['x_proj'], ds.n_pts)
        y_invalid_stats = calculate_invalid_stats(result['y_proj'], ds.n_pts)
        
        pure_counts[pair_idx] = b2d.pure_count
        
        # Build projection pair data based on validation mode
        pair_data = {
            'binner': b2d.binner,
            'xwt': result['xwt'],
            'ywt': result['ywt'],
            'x_transforms': result['x_transforms'],
            'y_transforms': result['y_transforms'],
            'pure_count': b2d.pure_count,
            'x_proj': result['x_proj'],
            'y_proj': result['y_proj'],
            'validation_attempts': result['attempt'],
            'validation_mode': result.get('validation_mode', '1d'),
            'x_invalid_stats': x_invalid_stats,
            'y_invalid_stats': y_invalid_stats
        }
        
        # Add mode-specific fields
        if result.get('validation_mode') == '1d':
            pair_data['x_occupied_bins'] = result.get('x_occupied', 0)
            pair_data['y_occupied_bins'] = result.get('y_occupied', 0)
        else:  # 2d mode
            pair_data['occupied_2d_bins'] = result.get('occupied_2d_bins', 0)
        
        # IMMEDIATELY WRITE TO DISK instead of storing in memory
        # Create filenames based on pair index (1-indexed)
        png_filename = f"pair_{pair_idx+1:02d}_pure_count_{pair_data['pure_count']:.0f}.png"
        json_filename = f"pair_{pair_idx+1:02d}_pure_count_{pair_data['pure_count']:.0f}.json"
        
        png_save_path = os.path.join(output_dir, png_filename)
        json_save_path = os.path.join(output_dir, json_filename)

        title = f"Class {current_class_original} ({group_label}) - 2D Projection Pair {pair_idx+1} with Transforms"

        # Set invalid statistics for plotting function
        plot_2d_projection._current_invalid_stats = {
            'x_invalid': pair_data.get('x_invalid_stats', {}),
            'y_invalid': pair_data.get('y_invalid_stats', {})
        }
        
        # Save plot
        plot_2d_projection(pair_data['binner'], class_map,
                         group_id_str_to_label, group_label_to_color, pair_data['xwt'], pair_data['ywt'],
                         pair_data['x_transforms'], pair_data['y_transforms'],
                         pair_data['pure_count'], pair_idx+1, title, png_save_path, current_class_mapped, features,
                         n_bins=n_bins,
                         x_occupied_bins=pair_data.get('x_occupied_bins'),
                         y_occupied_bins=pair_data.get('y_occupied_bins'),
                         occupied_2d_bins=pair_data.get('occupied_2d_bins'))
        
        # Export JSON data
        export_bin_data_to_json(pair_data['binner'], pair_data['xwt'], pair_data['ywt'], 
                               pair_data['x_transforms'], pair_data['y_transforms'],
                               features, current_class_mapped, class_map, id_to_name,
                               pair_data['pure_count'], n_bins, json_save_path)

        # Optionally export per-point 2D points
        if args.export_point_json:
            points_filename = f"pair_{pair_idx+1:02d}_points.json"
            points_path = os.path.join(points_dir, points_filename)
            export_pair_points_json(pair_data['x_proj'], pair_data['y_proj'],
                                    y_mapped, class_map, id_to_name,
                                    points_path, exclude_normal=args.exclude_normal)
        
        print(f"    Saved pair {pair_idx+1} to disk: {png_filename}")
        
        # Write detailed log entry for this pair immediately (while we still have the binner)
        log_file = os.path.join(output_dir, "detailed_analysis.log")
        
        # Create log file with header on first pair
        if pair_idx == 0:
            with open(log_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("CHIRP 1D PROJECTION PAIRS DETAILED ANALYSIS LOG (WITH TRANSFORMATIONS)\n")
                f.write("=" * 80 + "\n\n")

                f.write("ANALYSIS OVERVIEW:\n")
                f.write("-" * 40 + "\n")
                f.write("This analysis generates pairs of 1D projections using best variables with 75% sparsity.\n")
                f.write("Sign flipping is integrated into the pairing process rather than projection generation.\n")
                f.write("Each base projection is systematically combined with sign-flipped variants.\n")
                f.write("RANDOM TRANSFORMATIONS are applied to each feature before projection.\n")
                f.write("Memory optimization: Each pair written to disk immediately after generation.\n\n")

                # Log available features
                f.write("AVAILABLE FEATURES:\n")
                f.write("-" * 40 + "\n")
                for i, feature in enumerate(features):
                    f.write(f"Feature {i}: {feature}\n")

                f.write("\n" + "=" * 80 + "\n\n")

                # Log 2D pair analysis header
                f.write("1D PROJECTION PAIRS ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Number of bins: {n_bins}\n")
                f.write(f"Validation mode: {args.validation_mode.upper()}\n\n")
        
        # Append this pair's details to the log
        with open(log_file, 'a') as f:
            # Calculate bin statistics from the binner
            binner = pair_data['binner']
            n_bins_check = binner.get_num_bins()
            total_bins = n_bins_check * n_bins_check
            pure_bins = 0
            total_points_in_pure_bins = 0

            for i in range(n_bins_check):
                for j in range(n_bins_check):
                    if binner.pure_count(i, j) > 0:
                        pure_bins += 1
                        total_points_in_pure_bins += binner.pure_count(i, j)

            f.write(f"\nPair {pair_idx+1}:\n")
            f.write(f"  Pure Count: {pair_data['pure_count']:.0f}\n")
            f.write(f"  Number of Bins: {n_bins_check} x {n_bins_check} = {total_bins}\n")
            f.write(f"  Number of Pure Bins: {pure_bins} ({pure_bins/total_bins*100:.1f}%)\n")
            if pure_bins > 0:
                f.write(f"  Average Points per Pure Bin: {pair_data['pure_count']/pure_bins:.1f}\n")
            else:
                f.write(f"  Average Points per Pure Bin: N/A (no pure bins)\n")
            
            # Include validation statistics if available
            if 'x_occupied_bins' in pair_data and 'y_occupied_bins' in pair_data:
                f.write(f"  X Bins Occupied: {pair_data['x_occupied_bins']}/{n_bins_check} ({pair_data['x_occupied_bins']/n_bins_check*100:.1f}%)\n")
                f.write(f"  Y Bins Occupied: {pair_data['y_occupied_bins']}/{n_bins_check} ({pair_data['y_occupied_bins']/n_bins_check*100:.1f}%)\n")
            elif 'occupied_2d_bins' in pair_data:
                f.write(f"  2D Bins Occupied: {pair_data['occupied_2d_bins']}/{total_bins} ({pair_data['occupied_2d_bins']/total_bins*100:.1f}%)\n")
                
            if 'validation_attempts' in pair_data:
                f.write(f"  Validation Attempts: {pair_data['validation_attempts']}\n")
            
            # Include invalid value statistics if available
            if 'x_invalid_stats' in pair_data:
                f.write(f"  X Invalid Values: {pair_data['x_invalid_stats']['total_invalid']}/{pair_data['x_invalid_stats']['total_points']} "
                        f"({pair_data['x_invalid_stats']['invalid_percentage']:.1f}%) - "
                        f"Fully: {pair_data['x_invalid_stats']['fully_invalid']}, "
                        f"Partially: {pair_data['x_invalid_stats']['partially_invalid']}\n")
            if 'y_invalid_stats' in pair_data:
                f.write(f"  Y Invalid Values: {pair_data['y_invalid_stats']['total_invalid']}/{pair_data['y_invalid_stats']['total_points']} "
                        f"({pair_data['y_invalid_stats']['invalid_percentage']:.1f}%) - "
                        f"Fully: {pair_data['y_invalid_stats']['fully_invalid']}, "
                        f"Partially: {pair_data['y_invalid_stats']['partially_invalid']}\n")
                
            f.write(f"  X Projection Weights: {pair_data['xwt'].tolist()}\n")
            f.write(f"  Y Projection Weights: {pair_data['ywt'].tolist()}\n")
            f.write(f"  X Transformations: {pair_data['x_transforms']}\n")
            f.write(f"  Y Transformations: {pair_data['y_transforms']}\n")

            # Build projection formula strings (no commas between terms)
            def _build_formula(weights, transforms):
                tlist = []
                for i, w in enumerate(weights):
                    fidx = abs(w)
                    sign = "+" if w >= 0 else "-"
                    fname = features[fidx] if fidx < len(features) else f"Unknown_{fidx}"
                    trans = transforms[i]
                    tlist.append(f"{sign}{trans}({fname})")
                return ''.join(tlist)

            x_formula = _build_formula(pair_data['xwt'], pair_data['x_transforms'])
            y_formula = _build_formula(pair_data['ywt'], pair_data['y_transforms'])

            f.write(f"  X Projection Formula: {x_formula}\n")

            # Show which features are used in each projection with transformations
            f.write(f"  X Projection Features:\n")
            for i, weight in enumerate(pair_data['xwt']):
                feature_idx = abs(weight)
                sign = "+" if weight >= 0 else "-"
                feature_name = features[feature_idx] if feature_idx < len(features) else f"Unknown_{feature_idx}"
                transform = pair_data['x_transforms'][i]
                f.write(f"    {i}: {sign}{transform}({feature_name}) (index {feature_idx})\n")

            f.write(f"  Y Projection Features:\n")
            for i, weight in enumerate(pair_data['ywt']):
                feature_idx = abs(weight)
                sign = "+" if weight >= 0 else "-"
                feature_name = features[feature_idx] if feature_idx < len(features) else f"Unknown_{feature_idx}"
                transform = pair_data['y_transforms'][i]
                f.write(f"    {i}: {sign}{transform}({feature_name}) (index {feature_idx})\n")
        
        # Store only minimal summary data for statistics (not the heavy arrays or binner)
        pair_summary = {
            'xwt': pair_data['xwt'],
            'ywt': pair_data['ywt'],
            'x_transforms': pair_data['x_transforms'],
            'y_transforms': pair_data['y_transforms'],
            'pure_count': pair_data['pure_count'],
            'validation_attempts': pair_data['validation_attempts'],
            'validation_mode': pair_data['validation_mode'],
            'x_invalid_stats': x_invalid_stats,
            'y_invalid_stats': y_invalid_stats
        }
        
        # Add mode-specific fields to summary
        if result.get('validation_mode') == '1d':
            pair_summary['x_occupied_bins'] = result.get('x_occupied', 0)
            pair_summary['y_occupied_bins'] = result.get('y_occupied', 0)
        else:  # 2d mode
            pair_summary['occupied_2d_bins'] = result.get('occupied_2d_bins', 0)
        
        pair_summaries.append(pair_summary)
        
        # Clear heavy data from memory immediately after writing to disk
        del pair_data
        del b2d
        del result
        
        # Log memory usage every 25 pairs to track memory growth during computation
        if (pair_idx + 1) % 25 == 0:
            log_memory_usage(f"after pair {pair_idx+1}", start_memory)

    # Processing completed - record timing
    processing_end_time = time.time()
    processing_duration = processing_end_time - processing_start_time
    print(f"Projection pair generation completed at: {time.strftime('%H:%M:%S')}")
    print(f"Processing time: {processing_duration:.2f} seconds ({processing_duration/60:.2f} minutes)")
    log_memory_usage("after projection pair generation", start_memory)

    # Ranking removed: keep generation order
    
    # Display cache performance statistics
    cache_stats = _computation_cache.get_cache_stats()
    print(f"\nCache Performance Statistics:")
    print(f"  Cache hits: {cache_stats['cache_hits']}")
    print(f"  Cache misses: {cache_stats['cache_misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Transform cache entries: {cache_stats['transform_cache_size']}")
    print(f"  Bounds cache entries: {cache_stats['bounds_cache_size']}")
    print(f"  Projection cache entries: {cache_stats['projection_cache_size']}")
    if cache_stats['hit_rate'] > 0:
        print(f"  Cache system provided significant performance benefits!")
    
    # Display current memory usage
    log_memory_usage("after cache stats", start_memory)

    # Note: Detailed logging was written incrementally during pair generation
    log_file = os.path.join(output_dir, "detailed_analysis.log")
    print(f"Analysis log saved: {log_file}")

    # Note: Plotting was done immediately during pair generation to save memory
    print(f"\nAll {len(pair_summaries)} pairs have been generated and saved to disk.")
    print(f"Files saved in: {output_dir}")

    print(f"Completed processing class {current_class_original} ({group_label})")
    print(f"Plots saved in: {output_dir}")
    
    # Validation summary statistics
    validation_mode = args.validation_mode
    
    validation_stats = {}
    validation_stats['first_attempt_success'] = sum(1 for p in pair_summaries if p.get('validation_attempts', 1) == 1)
    validation_stats['retry_success'] = sum(1 for p in pair_summaries if p.get('validation_attempts', 1) > 1)
    
    # Add mode-specific statistics
    if validation_mode == '1d':
        validation_stats['avg_x_occupancy'] = np.mean([p.get('x_occupied_bins', 0) for p in pair_summaries]) / n_bins * 100
        validation_stats['avg_y_occupancy'] = np.mean([p.get('y_occupied_bins', 0) for p in pair_summaries]) / n_bins * 100
    else:  # 2d mode
        total_2d_bins = n_bins * n_bins
        validation_stats['avg_2d_occupancy'] = np.mean([p.get('occupied_2d_bins', 0) for p in pair_summaries]) / total_2d_bins * 100
        validation_stats['min_2d_occupancy'] = np.min([p.get('occupied_2d_bins', 0) for p in pair_summaries]) / total_2d_bins * 100 if pair_summaries else 0
        validation_stats['max_2d_occupancy'] = np.max([p.get('occupied_2d_bins', 0) for p in pair_summaries]) / total_2d_bins * 100 if pair_summaries else 0
    
    # Invalid value summary statistics
    if pair_summaries and 'x_invalid_stats' in pair_summaries[0]:
        invalid_stats = {
            'avg_x_invalid_percentage': np.mean([p.get('x_invalid_stats', {}).get('invalid_percentage', 0) for p in pair_summaries]),
            'avg_y_invalid_percentage': np.mean([p.get('y_invalid_stats', {}).get('invalid_percentage', 0) for p in pair_summaries]),
            'max_x_invalid_percentage': np.max([p.get('x_invalid_stats', {}).get('invalid_percentage', 0) for p in pair_summaries]),
            'max_y_invalid_percentage': np.max([p.get('y_invalid_stats', {}).get('invalid_percentage', 0) for p in pair_summaries]),
            'total_x_invalid': sum([p.get('x_invalid_stats', {}).get('total_invalid', 0) for p in pair_summaries]),
            'total_y_invalid': sum([p.get('y_invalid_stats', {}).get('total_invalid', 0) for p in pair_summaries])
        }
    
    print(f"\nValidation Statistics ({validation_mode.upper()} mode):")
    print(f"  First attempt success: {validation_stats['first_attempt_success']}/{len(pair_summaries)}")
    print(f"  Required retries: {validation_stats['retry_success']}/{len(pair_summaries)}")
    
    if validation_mode == '1d':
        print(f"  Average X bin occupancy: {validation_stats['avg_x_occupancy']:.1f}%")
        print(f"  Average Y bin occupancy: {validation_stats['avg_y_occupancy']:.1f}%")
    else:
        print(f"  Average 2D bin occupancy: {validation_stats['avg_2d_occupancy']:.1f}%")
        print(f"  Minimum 2D bin occupancy: {validation_stats['min_2d_occupancy']:.1f}%")
        print(f"  Maximum 2D bin occupancy: {validation_stats['max_2d_occupancy']:.1f}%")
    
    if pair_summaries and 'x_invalid_stats' in pair_summaries[0]:
        print(f"\nInvalid Value Statistics:")
        print(f"  Average X invalid values: {invalid_stats['avg_x_invalid_percentage']:.1f}%")
        print(f"  Average Y invalid values: {invalid_stats['avg_y_invalid_percentage']:.1f}%")
        print(f"  Maximum X invalid values: {invalid_stats['max_x_invalid_percentage']:.1f}%")
        print(f"  Maximum Y invalid values: {invalid_stats['max_y_invalid_percentage']:.1f}%")
        print(f"  Total X invalid points: {invalid_stats['total_x_invalid']:,}")
        print(f"  Total Y invalid points: {invalid_stats['total_y_invalid']:,}")
    
    # Final cache statistics and cleanup
    final_cache_stats = _computation_cache.get_cache_stats()
    print(f"\nFinal Cache Statistics:")
    print(f"  Total cache entries: {final_cache_stats['transform_cache_size'] + final_cache_stats['bounds_cache_size'] + final_cache_stats['projection_cache_size']}")
    print(f"  Memory savings achieved through {final_cache_stats['cache_hits']} cache hits")
    
    # Display final memory usage before summary
    log_memory_usage("final (before summary)", start_memory)
    
    # Optionally clear cache to free memory (uncomment if needed)
    # _computation_cache.clear_cache()
    # print("Cache cleared to free memory")

    # Calculate total execution time and final memory usage
    total_end_time = time.time()
    total_duration = total_end_time - start_time
    final_memory = get_memory_usage()
    memory_change = final_memory - start_memory
    
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print(f"  EXECUTION TIME AND MEMORY SUMMARY")
    print(f"{'='*80}")
    print(f"{'='*80}")
    print(f"\n>>> RUNTIME PERFORMANCE <<<")
    print(f"  Total execution time:        {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"  Projection generation & I/O: {processing_duration:.2f} seconds ({processing_duration/60:.2f} minutes)")
    print(f"                               (includes plot generation and saving to disk)")
    print(f"  Other operations:            {(total_duration - processing_duration):.2f} seconds")
    print(f"  Average time per pair:       {processing_duration/m_pairs:.2f} seconds" if m_pairs > 0 else "")
    print(f"\n>>> MEMORY USAGE <<<")
    print(f"  Initial: {format_memory(start_memory)}")
    print(f"  Final:   {format_memory(final_memory)}")
    print(f"  Change:  {format_memory(memory_change)}")
    print(f"\n>>> MEMORY OPTIMIZATION <<<")
    print(f"  Each pair was written to disk immediately after generation")
    print(f"  Only lightweight summaries ({len(pair_summaries)} pairs) kept in memory")
    print(f"\n{'='*80}")
    print(f"{'='*80}")

    # Append runtime information to log file
    with open(log_file, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("=" * 80 + "\n")
        f.write("  EXECUTION TIME AND MEMORY SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write("=" * 80 + "\n")
        f.write(f"\n>>> RUNTIME PERFORMANCE <<<\n")
        f.write(f"  Total execution time:        {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)\n")
        f.write(f"  Projection generation & I/O: {processing_duration:.2f} seconds ({processing_duration/60:.2f} minutes)\n")
        f.write(f"                               (includes plot generation and saving to disk)\n")
        f.write(f"  Other operations:            {(total_duration - processing_duration):.2f} seconds\n")
        if m_pairs > 0:
            f.write(f"  Average time per pair:       {processing_duration/m_pairs:.2f} seconds\n")
        f.write(f"\n>>> MEMORY USAGE <<<\n")
        f.write(f"  Initial: {format_memory(start_memory)}\n")
        f.write(f"  Final:   {format_memory(final_memory)}\n")
        f.write(f"  Change:  {format_memory(memory_change)}\n")
        f.write(f"\n>>> MEMORY OPTIMIZATION <<<\n")
        f.write(f"  Each pair was written to disk immediately after generation\n")
        f.write(f"  Only lightweight summaries ({len(pair_summaries)} pairs) kept in memory\n")
        f.write(f"\n>>> CACHE STATISTICS <<<\n")
        f.write(f"  Cache hits: {final_cache_stats['cache_hits']}\n")
        f.write(f"  Cache misses: {final_cache_stats['cache_misses']}\n")
        f.write(f"  Hit rate: {final_cache_stats['hit_rate']:.1%}\n")
        f.write(f"  Transform cache entries: {final_cache_stats['transform_cache_size']}\n")
        f.write(f"  Bounds cache entries: {final_cache_stats['bounds_cache_size']}\n")
        f.write(f"  Projection cache entries: {final_cache_stats['projection_cache_size']}\n")
        f.write(f"\n>>> VALIDATION STATISTICS ({validation_mode.upper()} mode) <<<\n")
        f.write(f"  First attempt success: {validation_stats['first_attempt_success']}/{len(pair_summaries)}\n")
        f.write(f"  Required retries: {validation_stats['retry_success']}/{len(pair_summaries)}\n")
        if validation_mode == '1d':
            f.write(f"  Average X bin occupancy: {validation_stats['avg_x_occupancy']:.1f}%\n")
            f.write(f"  Average Y bin occupancy: {validation_stats['avg_y_occupancy']:.1f}%\n")
        else:
            f.write(f"  Average 2D bin occupancy: {validation_stats['avg_2d_occupancy']:.1f}%\n")
            f.write(f"  Minimum 2D bin occupancy: {validation_stats['min_2d_occupancy']:.1f}%\n")
            f.write(f"  Maximum 2D bin occupancy: {validation_stats['max_2d_occupancy']:.1f}%\n")
        if pair_summaries and 'x_invalid_stats' in pair_summaries[0]:
            f.write(f"\n>>> INVALID VALUE STATISTICS <<<\n")
            f.write(f"  Average X invalid values: {invalid_stats['avg_x_invalid_percentage']:.1f}%\n")
            f.write(f"  Average Y invalid values: {invalid_stats['avg_y_invalid_percentage']:.1f}%\n")
            f.write(f"  Maximum X invalid values: {invalid_stats['max_x_invalid_percentage']:.1f}%\n")
            f.write(f"  Maximum Y invalid values: {invalid_stats['max_y_invalid_percentage']:.1f}%\n")
            f.write(f"  Total X invalid points: {invalid_stats['total_x_invalid']:,}\n")
            f.write(f"  Total Y invalid points: {invalid_stats['total_y_invalid']:,}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"{'='*80}\n")

    print("\nProcessing complete. 1D projection pairs with random transformations have been generated.")
    print("Performance optimizations through caching have been applied automatically.")

if __name__ == "__main__":
    main()
