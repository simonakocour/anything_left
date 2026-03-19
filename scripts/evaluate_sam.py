#!/usr/bin/env python3
"""Evaluate SAM segmentation masks for Remove360 scenes."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remove360.evaluation.sam import evaluate_sam_scene
from remove360.utils.io import save_json


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAM segmentation masks (Remove360)"
    )
    parser.add_argument(
        "scene_dirs",
        nargs="+",
        type=str,
        help="Paths to scene directories",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: scene_dir/evaluation_sam)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar",
    )
    args = parser.parse_args()

    for scene_dir in args.scene_dirs:
        path = Path(scene_dir)
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping.")
            continue
        out_dir = Path(args.output_dir) if args.output_dir else path / "evaluation_sam"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            matched = evaluate_sam_scene(str(path), verbose=not args.quiet)
            out_path = out_dir / "results.json"
            save_json(matched, str(out_path))
            sim_vals = [v["sim_sam"] for v in matched.values() if isinstance(v, dict)]
            if sim_vals:
                import numpy as np
                print(f"  sim_SAM (↑): {np.mean(sim_vals):.3f} ± {np.std(sim_vals):.3f}")
            print(f"Saved: {out_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
