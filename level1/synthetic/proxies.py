"""LD / FD / GC distribution proxies (adapted from simplified_dino_core/proxies.py)."""

from __future__ import annotations

import numpy as np
import torch


@torch.inference_mode()
def distribution_proxies(
    features: torch.Tensor,
    labels_np: np.ndarray,
    labeled_indices: np.ndarray,
    reference_class_frequencies: np.ndarray,
    num_classes: int,
    chunk_size: int,
) -> tuple[float, float, float]:
    """Compute LD / FD / GC control proxies."""
    n = features.shape[0]
    labeled_class_frequencies = np.bincount(
        labels_np[labeled_indices], minlength=num_classes
    ).astype(np.float64)
    labeled_class_frequencies /= len(labeled_indices)
    label_discrepancy = float(
        0.5 * np.abs(labeled_class_frequencies - reference_class_frequencies).sum()
    )

    unique_labeled = np.unique(labeled_indices)
    unique_idx = torch.from_numpy(unique_labeled.astype(np.int64)).to(features.device)
    labeled_features = features[unique_idx]

    fd_sum = torch.zeros((), device=features.device, dtype=torch.float64)
    for begin in range(0, n, chunk_size):
        end = min(begin + chunk_size, n)
        max_similarity = torch.full(
            (end - begin,), -float("inf"), device=features.device
        )
        for s_begin in range(0, len(labeled_features), chunk_size):
            similarity = (
                features[begin:end]
                @ labeled_features[s_begin : s_begin + chunk_size].T
            )
            max_similarity = torch.maximum(
                max_similarity, similarity.max(dim=1).values
            )
        fd_sum += (1 - max_similarity.float().clamp(max=1)).clamp_min(0).double().sum()
    feature_discrepancy = float(fd_sum / n)

    gc_sum = torch.zeros((), device=features.device, dtype=torch.float64)
    for begin in range(0, len(labeled_features), chunk_size):
        end = min(begin + chunk_size, len(labeled_features))
        max_similarity = torch.full(
            (end - begin,), -float("inf"), device=features.device
        )
        for s_begin in range(0, len(labeled_features), chunk_size):
            similarity = (
                labeled_features[begin:end]
                @ labeled_features[s_begin : s_begin + chunk_size].T
            )
            overlap_begin = max(begin, s_begin)
            overlap_end = min(end, s_begin + similarity.shape[1])
            if overlap_begin < overlap_end:
                rows = torch.arange(
                    overlap_begin - begin,
                    overlap_end - begin,
                    device=features.device,
                )
                cols = torch.arange(
                    overlap_begin - s_begin,
                    overlap_end - s_begin,
                    device=features.device,
                )
                similarity[rows, cols] = -float("inf")
            max_similarity = torch.maximum(
                max_similarity, similarity.max(dim=1).values
            )
        gc_sum += (1 - max_similarity.float().clamp(max=1)).clamp_min(0).double().sum()
    geometric_coverage = float(gc_sum / len(labeled_features))
    return label_discrepancy, feature_discrepancy, geometric_coverage
