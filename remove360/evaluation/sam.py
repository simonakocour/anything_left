"""SAM segmentation evaluation: IoU matching between GT and predicted segments."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from remove360.utils.io import load_binary_mask, load_sam_data, save_json


def _calculate_iou_np(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """IoU for numpy arrays."""
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return float(intersection / union) if union != 0 else 0.0


def _resize_mask(mask: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Resize mask with nearest-neighbor interpolation."""
    return cv2.resize(
        mask.astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )


def evaluate_sam_scene(
    scene_dir: str,
    gt_json_name: str = "merged.json",
    gt_binary_subdir: str = "masks",
    sam_subdir: str = "sam/after",
    overlap_threshold: float = 0.10,
    verbose: bool = True,
) -> Dict[str, List[dict]]:
    """Evaluate SAM segmentation masks against ground truth for one scene.

    Performs Hungarian matching on IoU matrix between GT and predicted segments,
    then records matched pairs.

    Expected structure:
        {scene_dir}/
            masks/merged.json or masks/{image_name}.json
            masks/{image_name}.png
            sam/after/merged.json or sam/after/{image_name}.json

    Args:
        scene_dir: Path to scene directory.
        gt_json_name: Filename of GT merged JSON.
        gt_binary_subdir: Subdir for GT binary masks.
        sam_subdir: Subdir for SAM output (merged.json).
        overlap_threshold: Min overlap (0.1) with object to consider a segment (paper).
        verbose: Print progress.

    Returns:
        Dict mapping image_name -> list of {gt_idx, sam_idx, iou}.
    """
    base = Path(scene_dir)
    gt_dir = base / gt_binary_subdir
    sam_dir = base / sam_subdir

    gt_data = load_sam_data(str(gt_dir))
    sam_data = load_sam_data(str(sam_dir))

    if not gt_data:
        raise FileNotFoundError(f"No GT SAM data in {gt_dir} (merged.json or *.json)")
    if not sam_data:
        raise FileNotFoundError(f"No SAM data in {sam_dir} (merged.json or *.json)")

    matched_segments = {}
    iterator = tqdm(gt_data, desc=base.name) if verbose else gt_data

    for image_name in iterator:
        if image_name not in sam_data:
            continue

        # Support both with and without .png extension
        gt_binary_path = gt_dir / image_name
        if not gt_binary_path.exists():
            gt_binary_path = gt_dir / f"{image_name}.png"
        if not gt_binary_path.exists():
            continue

        gt_mask_binary = load_binary_mask(str(gt_binary_path))
        h, w = gt_mask_binary.shape

        gt_segments = gt_data[image_name]
        sam_segments = sam_data[image_name]

        def _filter_segments(segments: List[dict]) -> List[np.ndarray]:
            masks = []
            for seg in segments:
                # Support both "mask" and "segmentation" (from run_sam_on_rgb.py)
                raw = seg.get("mask") if "mask" in seg else seg.get("segmentation")
                if raw is None:
                    continue
                mask = np.array(raw)
                resized = _resize_mask(mask, h, w).astype(bool)
                gt_bin = gt_mask_binary.astype(bool)
                inter = (resized & gt_bin).sum()
                union = (resized | gt_bin).sum()
                iou = inter / union if union > 0 else 0.0
                if iou >= overlap_threshold:
                    masks.append(resized.astype(np.uint8))
            return masks

        gt_masks = _filter_segments(gt_segments)
        sam_masks = _filter_segments(sam_segments)

        if not gt_masks or not sam_masks:
            continue

        gt_stack = np.stack(gt_masks)
        sam_stack = np.stack(sam_masks)
        gt_torch = torch.from_numpy(gt_stack).float()
        sam_torch = torch.from_numpy(sam_stack).float()

        n_gt, n_sam = gt_torch.shape[0], sam_torch.shape[0]
        iou_matrix = np.zeros((n_gt, n_sam))
        for i in range(n_gt):
            for j in range(n_sam):
                iou_matrix[i, j] = _calculate_iou_np(
                    gt_torch[i].int().numpy(),
                    sam_torch[j].int().numpy(),
                )

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # maximize

        matches = []
        matched_iou_sum = 0.0
        for gt_idx, sam_idx in zip(row_ind, col_ind):
            iou_val = iou_matrix[gt_idx, sam_idx]
            if iou_val > 0:
                matches.append({
                    "gt_idx": int(gt_idx),
                    "sam_idx": int(sam_idx),
                    "iou": float(iou_val),
                })
                matched_iou_sum += iou_val

        # Paper Eq.3: sim_SAM = sum(IoU(matched)) / max(N,M)
        n_gt, n_sam = len(gt_masks), len(sam_masks)
        sim_sam = matched_iou_sum / max(n_gt, n_sam) if (n_gt or n_sam) else 0.0
        matched_segments[image_name] = {
            "matches": matches,
            "sim_sam": float(sim_sam),
            "n_gt": n_gt,
            "n_sam": n_sam,
        }

    return matched_segments
