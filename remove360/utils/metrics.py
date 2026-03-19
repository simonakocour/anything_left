"""Evaluation metrics for Remove360."""

from typing import Optional, Union

import numpy as np
import torch


def calculate_iou(
    mask1: Union[np.ndarray, torch.Tensor],
    mask2: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute Intersection over Union between two binary masks.

    Args:
        mask1: First binary mask.
        mask2: Second binary mask.

    Returns:
        IoU score in [0, 1].
    """
    if isinstance(mask1, np.ndarray):
        mask1 = torch.from_numpy(mask1)
    if isinstance(mask2, np.ndarray):
        mask2 = torch.from_numpy(mask2)

    mask1 = (mask1 > 0).long()
    mask2 = (mask2 > 0).long()

    intersection = torch.logical_and(mask1, mask2).sum().item()
    union = torch.logical_or(mask1, mask2).sum().item()
    return intersection / union if union > 0 else 0.0


def calculate_accuracy(
    reference: Union[np.ndarray, torch.Tensor],
    prediction: Union[np.ndarray, torch.Tensor],
    target: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """Compute per-pixel accuracy within the reference mask region.

    Args:
        reference: Binary mask defining the region of interest.
        prediction: Predicted mask.
        target: Ground truth mask (defaults to reference for before/after comparison).

    Returns:
        Accuracy in [0, 1].
    """
    if isinstance(reference, np.ndarray):
        reference = torch.from_numpy(reference)
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction)

    if target is None:
        target = reference
    elif isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    prediction = prediction * reference
    correct = ((prediction == target) & (reference > 0)).sum().item()
    total = (reference > 0).sum().item()
    return correct / total if total > 0 else 0.0
