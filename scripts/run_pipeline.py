#!/usr/bin/env python3
"""Run Remove360 pipeline: SAM (if needed) → depth diff (if needed) → evaluation.

Skips steps when outputs already exist. Input: path to scene dir with RGBs.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Paths — edit if needed
DEFAULT_SAM_CHECKPOINT = str(ROOT / "checkpoints" / "sam_vit_h_4b8939.pth")
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Run Remove360 pipeline (skips steps when output exists)"
    )
    parser.add_argument("scene_dir", help="Scene directory (path with RGBs)")
    parser.add_argument(
        "--images-before",
        default=None,
        help="GT before-removal images (default: scene_dir/images/before)",
    )
    parser.add_argument(
        "--images-after",
        default=None,
        help="GT after-removal images (default: scene_dir/images/after)",
    )
    parser.add_argument(
        "--rgb-before",
        default=None,
        help="Method renders before removal (default: scene_dir/rgb/before)",
    )
    parser.add_argument(
        "--rgb-after",
        default=None,
        help="Method renders after removal (default: scene_dir/rgb/after)",
    )
    parser.add_argument(
        "--depth-before",
        default=None,
        help="Depth before (default: scene_dir/depth/before)",
    )
    parser.add_argument(
        "--depth-after",
        default=None,
        help="Depth after (default: scene_dir/depth/after)",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=DEFAULT_SAM_CHECKPOINT,
        help="SAM checkpoint path",
    )
    parser.add_argument("--no-skip", action="store_true", help="Recompute all (no skip)")
    parser.add_argument("--skip-sam", action="store_true", help="Skip SAM (assume done)")
    parser.add_argument("--skip-depth-diff", action="store_true", help="Skip depth diff")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    args = parser.parse_args()

    os.chdir(ROOT)
    scene = Path(args.scene_dir).resolve()
    force = args.no_skip

    def run(cmd, desc):
        print(f"\n--- {desc} ---")
        r = subprocess.run(cmd, shell=True)
        if r.returncode != 0:
            sys.exit(r.returncode)

    images_before = Path(args.images_before or str(scene / "images" / "before"))
    images_after = Path(args.images_after or str(scene / "images" / "after"))
    rgb_before = Path(args.rgb_before or str(scene / "rgb" / "before"))
    rgb_after = Path(args.rgb_after or str(scene / "rgb" / "after"))
    depth_before = Path(args.depth_before or str(scene / "depth" / "before"))
    depth_after = Path(args.depth_after or str(scene / "depth" / "after"))

    sam_out = scene / "sam" / "after"
    depth_diff_out = scene / "depth_diff"

    # Step 1: SAM on rgb_after (method's after-removal renders)
    n_after = len(list(rgb_after.glob("*.png")) + list(rgb_after.glob("*.jpg"))) if rgb_after.exists() else 0
    n_sam_json = len(list(sam_out.glob("*.json"))) if sam_out.exists() else 0
    if not args.skip_sam and rgb_after.exists():
        if force or n_sam_json < n_after:
            run(
                f'python scripts/run_sam_on_rgb.py "{rgb_after}" -o "{scene / "sam" / "after"}" '
                f'--checkpoint "{args.sam_checkpoint}"' + (" --no-skip" if force else ""),
                "SAM on rgb_after",
            )
        else:
            print(f"\n--- Skip SAM on rgb/after ({n_sam_json} JSONs exist) ---")

    # Step 2: SAM on GT images (for sim_SAM) — images/before are GT before-removal
    masks_dir = scene / "masks"
    n_gt_imgs = len(list(images_before.glob("*.png")) + list(images_before.glob("*.jpg"))) if images_before.exists() else 0
    n_gt_json = len(list(masks_dir.glob("*.json"))) if masks_dir.exists() else 0
    if not args.skip_sam and images_before.exists() and (force or n_gt_json < n_gt_imgs):
        run(
            f'python scripts/run_sam_on_rgb.py "{images_before}" -o "{masks_dir}" '
            f'--checkpoint "{args.sam_checkpoint}"' + (" --no-skip" if force else ""),
            "SAM on GT images (images/before)",
        )
    elif images_before.exists() and n_gt_json >= n_gt_imgs:
        print(f"\n--- Skip SAM on GT ({n_gt_json} JSONs exist) ---")

    # Step 3: Depth diff
    if not args.skip_depth_diff and depth_before.exists() and depth_after.exists():
        existing = list(depth_diff_out.glob("*_final_threshold.png")) if depth_diff_out.exists() else []
        if force or len(existing) == 0:
            run(
                f'python scripts/compute_depth_diff.py "{depth_before}" "{depth_after}" '
                f'-o "{depth_diff_out}"' + (" --no-skip" if force else ""),
                "Depth diff",
            )
        else:
            print(f"\n--- Skip depth diff ({len(existing)} masks exist) ---")

    # Step 4: Evaluation
    if not args.skip_eval:
        print(f"\n--- Evaluation ---")
        run(f'python scripts/evaluate_semantic.py "{scene}"', "Semantic (IoU_drop, acc_seg)")
        run(f'python scripts/evaluate_depth.py "{scene}"', "Depth (acc_Δdepth)")
        run(f'python scripts/evaluate_sam.py "{scene}"', "SAM (sim_SAM)")


if __name__ == "__main__":
    main()
