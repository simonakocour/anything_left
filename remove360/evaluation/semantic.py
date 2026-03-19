"""Semantic evaluation: measure GSAM mask residuals after object removal."""

import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from remove360.utils.io import load_mask, save_json
from remove360.utils.metrics import calculate_iou, calculate_accuracy


def _resize_to_largest(*masks: np.ndarray) -> List[torch.Tensor]:
    """Resize all masks to the largest dimensions among them."""
    max_h = max(m.shape[0] for m in masks)
    max_w = max(m.shape[1] for m in masks)
    resized = []
    for mask in masks:
        tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        tensor = F.interpolate(tensor, size=(max_h, max_w), mode="nearest")
        resized.append((tensor.squeeze() > 0).long())
    return resized


def evaluate_semantic_scene(
    scene_dir: str,
    gsam_after_subdir: str = "gsam2/after/mask",
    gsam_before_subdir: str = "gsam2/before/mask",
    gt_subdir: str = "masks",
    output_subdir: str = "evaluation_semantic",
    verbose: bool = True,
) -> List[dict]:
    """Evaluate semantic (GSAM) residuals for a single scene.

    Compares GSAM masks before vs after removal. Ideal removal: IoU_after → 0,
    Acc_after → 1 (no object detected in removal region).

    Expected structure:
        {scene_dir}/
            gsam2/after/mask/     # masks after removal
            gsam2/before/mask/   # masks before removal
            masks/              # ground truth object masks (same as HF)

    Args:
        scene_dir: Path to scene directory.
        gsam_after_subdir: Subdir for after-removal masks.
        gsam_before_subdir: Subdir for before-removal masks.
        gt_subdir: Ground truth mask directory.
        output_subdir: Output directory for JSON.
        verbose: Print per-image results.

    Returns:
        List of per-image results.
    """
    base = Path(scene_dir)
    after_dir = base / gsam_after_subdir
    before_dir = base / gsam_before_subdir
    gt_dir = base / gt_subdir
    out_dir = base / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    image_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])
    results = []

    for img_name in image_files:
        ref_path = gt_dir / img_name
        before_path = before_dir / img_name
        after_path = after_dir / img_name

        ref_mask = load_mask(str(ref_path))
        if ref_mask is None:
            if verbose:
                print(f"  Reference missing: {img_name}, skipping.")
            continue

        before_mask = load_mask(str(before_path))
        after_mask = load_mask(str(after_path))
        if before_mask is None:
            before_mask = np.zeros_like(ref_mask)
        if after_mask is None:
            after_mask = np.zeros_like(ref_mask)

        ref_t, before_t, after_t = _resize_to_largest(ref_mask, before_mask, after_mask)

        iou_before = calculate_iou(ref_t, before_t)
        iou_after = calculate_iou(ref_t, after_t)
        acc_before = calculate_accuracy(ref_t, before_t, ref_t)
        acc_after = calculate_accuracy(ref_t, after_t, torch.zeros_like(ref_t))

        # Paper Eq.1: IoU_drop = IoU_pre - IoU_post (higher = better removal)
        iou_drop = float(iou_before - iou_after)

        entry = {
            "image_name": img_name,
            "iou_before": float(iou_before),
            "iou_after": float(iou_after),
            "iou_drop": iou_drop,
            "acc_before": float(acc_before),
            "acc_after": float(acc_after),
            "acc_diff": float(acc_after - acc_before),
        }
        results.append(entry)

        if verbose:
            print(f"  {img_name}: IoU_drop={iou_drop:.4f} Acc_diff={entry['acc_diff']:.4f}")

    save_json(results, str(out_dir / "evaluation.json"))
    return results


def summarize_semantic_results(
    results: List[dict],
    label: str = "SUMMARY",
) -> dict:
    """Compute and optionally print summary statistics."""
    if not results:
        if label:
            print(f"{label}: No data")
        return {}

    n = len(results)
    iou_before = np.mean([r["iou_before"] for r in results])
    iou_after = np.mean([r["iou_after"] for r in results])
    iou_drop = np.mean([r["iou_drop"] for r in results])
    acc_before = np.mean([r["acc_before"] for r in results])
    acc_after = np.mean([r["acc_after"] for r in results])
    acc_diff = np.mean([r["acc_diff"] for r in results])
    # Paper Eq.2: acc_seg = fraction of images with IoU_post < threshold
    acc_seg_30 = sum(1 for r in results if r["iou_after"] < 0.30) / n
    acc_seg_50 = sum(1 for r in results if r["iou_after"] < 0.50) / n

    summary = {
        "n_images": n,
        "mean_iou_before": float(iou_before),
        "mean_iou_after": float(iou_after),
        "mean_iou_drop": float(iou_drop),
        "mean_acc_before": float(acc_before),
        "mean_acc_after": float(acc_after),
        "mean_acc_diff": float(acc_diff),
        "acc_seg_30": float(acc_seg_30),
        "acc_seg_50": float(acc_seg_50),
    }

    if label:
        print(f"\n{label}:")
        print(f"  N images:        {n}")
        print(f"  IoU_drop (↑):    {iou_drop:.4f}  (mean IoU_pre - IoU_post)")
        print(f"  acc_seg@0.3 (↑): {acc_seg_30:.2%}  (frac. IoU_post < 0.3)")
        print(f"  acc_seg@0.5 (↑): {acc_seg_50:.2%}  (frac. IoU_post < 0.5)")

    return summary
