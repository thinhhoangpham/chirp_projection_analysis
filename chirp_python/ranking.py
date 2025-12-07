import math
from typing import Dict, List, Tuple
import numpy as np
from chirp_python.sorter import Sorter


def compute_pair_scores(binner, target_idx: int) -> Dict[str, float]:
    """Compute separation-focused scores for a 2D binner grid.

    Returns a dict with keys:
      - f1max: Max F1 across purity thresholds over bins
      - auc: ROC AUC using bin purity as score
      - bp: Balanced purity (coverage-weighted purity for target mass)
      - jsd: Jensen–Shannon divergence between target vs non-target distributions
      - purity_weighted: Mean bin purity weighted by bin count
      - coverage_p70: Fraction of target mass in bins with purity >= 0.70
      - target_total, non_target_total, occupied_bins: diagnostics
    """

    bins = binner.get_bins()
    n_bins = binner.get_num_bins()

    tgt_total = 0
    non_total = 0
    per_bin = []  # (purity, tgt_count, non_count, total_count)

    for i in range(n_bins):
        for j in range(n_bins):
            b = bins[i][j]
            if b.count <= 0:
                continue
            tgt = int(b.class_counts[target_idx])
            non = int(b.count - tgt)
            p = tgt / b.count if b.count > 0 else 0.0
            per_bin.append((p, tgt, non, int(b.count)))
            tgt_total += tgt
            non_total += non

    total_points = tgt_total + non_total

    # Balanced Purity: E_bin[purity | target]
    bp = 0.0
    if tgt_total > 0:
        s = 0.0
        for (p, tgt, _non, cnt) in per_bin:
            if cnt > 0:
                s += (tgt * p)  # == tgt^2 / cnt
        bp = s / tgt_total

    # Weighted mean purity over occupied bins
    purity_weighted = 0.0
    if total_points > 0:
        s = 0.0
        for (p, _tgt, _non, cnt) in per_bin:
            s += p * cnt
        purity_weighted = s / total_points

    # Coverage at 0.70 purity
    coverage_p70 = 0.0
    if tgt_total > 0:
        s = 0.0
        for (p, tgt, _non, _cnt) in per_bin:
            if p >= 0.70:
                s += tgt
        coverage_p70 = s / tgt_total

    # ROC AUC using bin purity as score
    auc = 0.5
    if tgt_total > 0 and non_total > 0 and per_bin:
        per_bin_sorted = sorted(per_bin, key=lambda x: x[0], reverse=True)
        cum_tgt = 0
        cum_non = 0
        prev_tpr = 0.0
        prev_fpr = 0.0
        auc = 0.0
        idx = 0
        while idx < len(per_bin_sorted):
            t = per_bin_sorted[idx][0]
            add_tgt = 0
            add_non = 0
            while idx < len(per_bin_sorted) and per_bin_sorted[idx][0] == t:
                add_tgt += per_bin_sorted[idx][1]
                add_non += per_bin_sorted[idx][2]
                idx += 1
            cum_tgt += add_tgt
            cum_non += add_non
            tpr = cum_tgt / tgt_total if tgt_total > 0 else 0.0
            fpr = cum_non / non_total if non_total > 0 else 0.0
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
            prev_tpr, prev_fpr = tpr, fpr

    # F1-max across thresholds (by walking purity from high to low)
    f1max = 0.0
    if tgt_total > 0 and per_bin:
        per_bin_sorted = sorted(per_bin, key=lambda x: x[0], reverse=True)
        cum_tgt = 0
        cum_non = 0
        idx = 0
        while idx < len(per_bin_sorted):
            t = per_bin_sorted[idx][0]
            add_tgt = 0
            add_non = 0
            while idx < len(per_bin_sorted) and per_bin_sorted[idx][0] == t:
                add_tgt += per_bin_sorted[idx][1]
                add_non += per_bin_sorted[idx][2]
                idx += 1
            cum_tgt += add_tgt
            cum_non += add_non
            tp = cum_tgt
            fp = cum_non
            fn = tgt_total - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if prec + rec > 0:
                f1 = 2.0 * prec * rec / (prec + rec)
                if f1 > f1max:
                    f1max = f1

    # Jensen–Shannon divergence between P(target|bin) and P(non|bin)
    jsd = 0.0
    if tgt_total > 0 and non_total > 0 and per_bin:
        P = [t / tgt_total for (_p, t, _n, _c) in per_bin]
        Q = [n / non_total for (_p, _t, n, _c) in per_bin]
        M = [(P[i] + Q[i]) / 2.0 for i in range(len(P))]

        def kl(A, B):
            s = 0.0
            for a, b in zip(A, B):
                if a > 0 and b > 0:
                    s += a * math.log(a / b)
            return s

        jsd = 0.5 * (kl(P, M) + kl(Q, M))
        jsd /= math.log(2.0)  # normalize to [0, 1]

    return {
        'bp': float(bp),
        'auc': float(auc),
        'f1max': float(f1max),
        'jsd': float(jsd),
        'purity_weighted': float(purity_weighted),
        'coverage_p70': float(coverage_p70),
        'target_total': int(tgt_total),
        'non_target_total': int(non_total),
        'occupied_bins': int(len(per_bin)),
    }


def compute_ranking_score(binner, target_idx: int, metric: str) -> float:
    """Compute only the requested ranking metric efficiently.

    Supported metrics:
      - 'f1'  : Max F1 across purity thresholds over bins
      - 'auc' : ROC AUC using bin purity as score
      - 'bp'  : Balanced purity, E[purity | target]
      - 'jsd' : Jensen–Shannon divergence between target and non-target distributions
      - 'sil' or 'silhouette': Mean silhouette over target points using bin centers
      - 'sil_all' or 'silhouette_all' (also 'sil_multi', 'silhouette_multi'):
            Multi-class silhouette averaged over all classes (weighted by counts)

    Note: 'pure' is computed outside via Bin2D.pure_count and Binner.pure_count.
    """

    bins = binner.get_bins()
    n_bins = binner.get_num_bins()

    tgt_total = 0
    non_total = 0
    per_bin = []  # (purity, tgt_count, non_count, total_count)

    for i in range(n_bins):
        for j in range(n_bins):
            b = bins[i][j]
            if b.count <= 0:
                continue
            tgt = int(b.class_counts[target_idx])
            non = int(b.count - tgt)
            p = tgt / b.count if b.count > 0 else 0.0
            per_bin.append((p, tgt, non, int(b.count)))
            tgt_total += tgt
            non_total += non

    if metric == 'bp':
        if tgt_total <= 0:
            return 0.0
        s = 0.0
        for (p, tgt, _non, cnt) in per_bin:
            if cnt > 0:
                s += (tgt * p)  # == tgt^2 / cnt
        return float(s / tgt_total)

    if metric == 'auc':
        if tgt_total <= 0 or non_total <= 0 or not per_bin:
            return 0.5
        per_bin_sorted = sorted(per_bin, key=lambda x: x[0], reverse=True)
        cum_tgt = 0
        cum_non = 0
        prev_tpr = 0.0
        prev_fpr = 0.0
        auc = 0.0
        idx = 0
        while idx < len(per_bin_sorted):
            t = per_bin_sorted[idx][0]
            add_tgt = 0
            add_non = 0
            while idx < len(per_bin_sorted) and per_bin_sorted[idx][0] == t:
                add_tgt += per_bin_sorted[idx][1]
                add_non += per_bin_sorted[idx][2]
                idx += 1
            cum_tgt += add_tgt
            cum_non += add_non
            tpr = cum_tgt / tgt_total if tgt_total > 0 else 0.0
            fpr = cum_non / non_total if non_total > 0 else 0.0
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
            prev_tpr, prev_fpr = tpr, fpr
        return float(auc)

    if metric == 'f1':
        if tgt_total <= 0 or not per_bin:
            return 0.0
        per_bin_sorted = sorted(per_bin, key=lambda x: x[0], reverse=True)
        cum_tgt = 0
        cum_non = 0
        f1max = 0.0
        idx = 0
        while idx < len(per_bin_sorted):
            t = per_bin_sorted[idx][0]
            add_tgt = 0
            add_non = 0
            while idx < len(per_bin_sorted) and per_bin_sorted[idx][0] == t:
                add_tgt += per_bin_sorted[idx][1]
                add_non += per_bin_sorted[idx][2]
                idx += 1
            cum_tgt += add_tgt
            cum_non += add_non
            tp = cum_tgt
            fp = cum_non
            fn = tgt_total - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if prec + rec > 0:
                f1 = 2.0 * prec * rec / (prec + rec)
                if f1 > f1max:
                    f1max = f1
        return float(f1max)

    if metric in ('sil', 'silhouette'):
        # Build weighted positions for target and non-target using bin centers
        tgt_positions = []
        tgt_weights = []
        non_positions = []
        non_weights = []
        for i in range(n_bins):
            for j in range(n_bins):
                b = bins[i][j]
                if b.count <= 0:
                    continue
                tgt = int(b.class_counts[target_idx])
                non = int(b.count - tgt)
                if tgt <= 0 and non <= 0:
                    continue
                # Bin center in normalized space
                pos = ((i + 0.5) / n_bins, (j + 0.5) / n_bins)
                if tgt > 0:
                    tgt_positions.append(pos)
                    tgt_weights.append(tgt)
                if non > 0:
                    non_positions.append(pos)
                    non_weights.append(non)

        if not tgt_weights or not non_weights:
            return 0.0

        tgt_positions = np.asarray(tgt_positions, dtype=np.float64)
        tgt_weights = np.asarray(tgt_weights, dtype=np.float64)
        non_positions = np.asarray(non_positions, dtype=np.float64)
        non_weights = np.asarray(non_weights, dtype=np.float64)

        N_t = float(tgt_weights.sum())
        N_o = float(non_weights.sum())
        if N_t <= 1.0 or N_o <= 0.0:
            return 0.0

        # For each target bin k, compute weighted average distance to all target and all non-target bins
        sil_sum = 0.0
        for k in range(tgt_positions.shape[0]):
            pos_k = tgt_positions[k]
            # Distances to all target bins
            d_t = np.sqrt(((tgt_positions - pos_k) ** 2).sum(axis=1))
            num_a = float(np.dot(tgt_weights, d_t))  # includes zeros for same-bin
            den_a = max(N_t - 1.0, 1e-12)
            a_k = num_a / den_a

            # Distances to all non-target bins
            d_o = np.sqrt(((non_positions - pos_k) ** 2).sum(axis=1))
            num_b = float(np.dot(non_weights, d_o))
            den_b = max(N_o, 1e-12)
            b_k = num_b / den_b

            m = max(a_k, b_k)
            s_k = (b_k - a_k) / m if m > 0 else 0.0
            sil_sum += tgt_weights[k] * s_k

        return float(sil_sum / N_t)

    if metric in ('sil_all', 'silhouette_all', 'sil_multi', 'silhouette_multi'):
        # Multi-class silhouette using bin centers; average over all classes weighted by counts
        chdr = binner.get_chdr()
        n_classes = getattr(chdr, 'n_classes', None)
        if n_classes is None:
            return 0.0

        class_positions = [[] for _ in range(n_classes)]
        class_weights = [[] for _ in range(n_classes)]
        for i in range(n_bins):
            for j in range(n_bins):
                b = bins[i][j]
                if b.count <= 0:
                    continue
                pos = ((i + 0.5) / n_bins, (j + 0.5) / n_bins)
                for c in range(n_classes):
                    w = int(b.class_counts[c])
                    if w > 0:
                        class_positions[c].append(pos)
                        class_weights[c].append(w)

        total_weight = 0.0
        for c in range(n_classes):
            if class_weights[c]:
                total_weight += float(sum(class_weights[c]))
        if total_weight <= 1.0:
            return 0.0

        sil_sum_total = 0.0
        # Convert lists to arrays once for each class to speed up inner loops
        class_pos_arr = [np.asarray(class_positions[c], dtype=np.float64) if class_positions[c] else None
                         for c in range(n_classes)]
        class_w_arr = [np.asarray(class_weights[c], dtype=np.float64) if class_weights[c] else None
                       for c in range(n_classes)]

        for c in range(n_classes):
            pos_c = class_pos_arr[c]
            w_c = class_w_arr[c]
            if pos_c is None:
                continue
            N_c = float(w_c.sum())
            if N_c <= 1.0:
                continue

            for k in range(pos_c.shape[0]):
                pos_k = pos_c[k]
                # Intra-class distance (exclude self by denominator adjustment)
                d_same = np.sqrt(((pos_c - pos_k) ** 2).sum(axis=1))
                num_a = float(np.dot(w_c, d_same))
                den_a = max(N_c - w_c[k], 1e-12)
                a_k = num_a / den_a

                # Inter-class: min average distance to another class
                b_k = None
                for d in range(n_classes):
                    if d == c:
                        continue
                    pos_d = class_pos_arr[d]
                    w_d = class_w_arr[d]
                    if pos_d is None:
                        continue
                    N_d = float(w_d.sum())
                    if N_d <= 0.0:
                        continue
                    d_other = np.sqrt(((pos_d - pos_k) ** 2).sum(axis=1))
                    avg_d = float(np.dot(w_d, d_other)) / N_d
                    if b_k is None or avg_d < b_k:
                        b_k = avg_d
                if b_k is None:
                    continue

                m = max(a_k, b_k)
                s_k = (b_k - a_k) / m if m > 0 else 0.0
                sil_sum_total += w_c[k] * s_k

        return float(sil_sum_total / total_weight)

    if metric == 'jsd':
        if tgt_total <= 0 or non_total <= 0 or not per_bin:
            return 0.0
        P = [t / tgt_total for (_p, t, _n, _c) in per_bin]
        Q = [n / non_total for (_p, _t, n, _c) in per_bin]
        M = [(P[i] + Q[i]) / 2.0 for i in range(len(P))]

        def kl(A, B):
            s = 0.0
            for a, b in zip(A, B):
                if a > 0 and b > 0:
                    s += a * math.log(a / b)
            return s

        jsd = 0.5 * (kl(P, M) + kl(Q, M))
        jsd /= math.log(2.0)
        return float(jsd)

    # Unknown metric
    return 0.0


def rank_pairs(projection_pairs: List[dict], metric: str, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rank a list of projection pairs by the requested metric.

    projection_pairs: list of dicts with at least keys:
      - 'binner': Binner instance
      - 'pure_count': int (used when metric == 'pure')

    Returns (ranking_indices, ranking_scores):
      - ranking_indices: np.ndarray of pair indices sorted by score desc
      - ranking_scores: np.ndarray of per-pair scores aligned with projection_pairs
    """
    n = len(projection_pairs)
    scores = np.zeros(n, dtype=float)
    for i, pair in enumerate(projection_pairs):
        if metric == 'pure':
            scores[i] = float(pair.get('pure_count', 0.0))
        else:
            scores[i] = compute_ranking_score(pair['binner'], target_idx, metric)
    indices = Sorter.descending_sort(scores)
    return indices, scores
