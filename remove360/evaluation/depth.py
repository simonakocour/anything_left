"""Depth evaluation: measure depth residuals after object removal."""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from remove360.utils.io import load_mask, save_json
from remove360.utils.metrics import calculate_iou, calculate_accuracy


def _match_size(ref: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize target mask to match reference dimensions."""
    if ref.shape != target.shape:
        target = F.interpolate(
            target.unsqueeze(0).unsqueeze(0).float(),
            size=ref.shape,
            mode="nearest-exact",
        ).squeeze().to(torch.uint8)
    return target


def evaluate_depth_scene(
    scene_dir: str,
    depth_subdir: str = "depth_diff",
    ref_subdir: str = "masks",
    output_subdir: str = "evaluation_depth_diff",
    suffix: str = "_final_threshold.png",
    verbose: bool = True,
) -> List[dict]:
    """Evaluate depth residuals for a single scene.

    Compares depth difference masks (after removal) against ground truth object
    masks. Lower IoU/accuracy indicates better removal (fewer depth residuals).

    Expected structure:
        {scene_dir}/
            {depth_subdir}/     # depth difference masks
            {ref_subdir}/       # ground truth object masks
            {output_subdir}/    # results written here

    Args:
        scene_dir: Path to scene directory.
        depth_subdir: Subpath to depth masks.
        ref_subdir: Subpath to reference masks.
        output_subdir: Subpath for output JSON.
        suffix: Filename suffix for depth masks.
        verbose: Print per-image results.

    Returns:
        List of per-image results with keys: image, iou, acc.
    """
    depth_path = Path(scene_dir) / depth_subdir
    ref_path = Path(scene_dir) / ref_subdir
    out_path = Path(scene_dir) / output_subdir
    out_path.mkdir(parents=True, exist_ok=True)

    if not depth_path.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_path}")

    image_names = [
        f.replace(suffix, "")
        for f in os.listdir(depth_path)
        if f.endswith(suffix)
    ]

    results = []
    for name in sorted(image_names):
        ref = load_mask(str(ref_path / f"{name}.png"))
        depth = load_mask(str(depth_path / f"{name}{suffix}"))

        if depth is None:
            if verbose:
                print(f"  Missing depth: {name}")
            continue

        if ref is None:
            ref = np.zeros_like(depth)

        ref_t = torch.from_numpy(ref)
        depth_t = torch.from_numpy(depth)
        depth_t = _match_size(ref_t, depth_t)

        ref_bin = ref_t > 0
        # Depth diff mask: 255 = no change, other = changed (paper Eq.4)
        depth_changed = depth_t != 255

        # Paper Eq.4: acc_Δdepth = (# object pixels with depth change) / (# object pixels)
        acc_delta_depth = calculate_accuracy(ref_bin, depth_changed, ref_bin)
        iou = calculate_iou(ref_bin, depth_changed)

        results.append({
            "image": name,
            "acc_delta_depth": float(acc_delta_depth),
            "iou": float(iou),
        })
        if verbose:
            print(f"  {name}: acc_Δdepth={acc_delta_depth:.3f} IoU={iou:.3f}")

    save_json(results, str(out_path / "depth_evaluation.json"))
    return results


def summarize_depth_results(
    results: List[dict],
    label: str = "SUMMARY",
) -> dict:
    """Compute and optionally print summary statistics."""
    if not results:
        if label:
            print(f"{label}: No data")
        return {}

    accs = [r["acc_delta_depth"] for r in results]
    ious = [r["iou"] for r in results]
    summary = {
        "mean_acc_delta_depth": float(np.mean(accs)),
        "std_acc_delta_depth": float(np.std(accs)),
        "mean_iou": float(np.mean(ious)),
        "n_images": len(results),
    }

    if label:
        print(f"\n{label}:")
        print(f"  acc_Δdepth (↑): {summary['mean_acc_delta_depth']:.3f} ± {summary['std_acc_delta_depth']:.3f}")
        print(f"  N images:       {summary['n_images']}")

    return summary
