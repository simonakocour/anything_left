#!/usr/bin/env python3
"""Evaluate depth residuals for Remove360 scenes."""

import argparse
import sys
from pathlib import Path

# Add project root to path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remove360.evaluation.depth import evaluate_depth_scene, summarize_depth_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate depth residuals after object removal (Remove360)"
    )
    parser.add_argument(
        "scene_dirs",
        nargs="+",
        type=str,
        help="Paths to scene directories",
    )
    parser.add_argument(
        "--depth-subdir",
        default="depth_diff",
        help="Subpath to depth masks",
    )
    parser.add_argument(
        "--ref-subdir",
        default="masks",
        help="Subpath to reference masks",
    )
    parser.add_argument(
        "--output-subdir",
        default="evaluation_depth_diff",
        help="Subpath for output JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image output",
    )
    args = parser.parse_args()

    all_results = []
    for scene_dir in args.scene_dirs:
        path = Path(scene_dir)
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping.")
            continue
        try:
            results = evaluate_depth_scene(
                str(path),
                depth_subdir=args.depth_subdir,
                ref_subdir=args.ref_subdir,
                output_subdir=args.output_subdir,
                verbose=not args.quiet,
            )
            all_results.extend(results)
            summarize_depth_results(results, label=path.name)
        except FileNotFoundError as e:
            print(f"Error: {e}")

    if all_results:
        summarize_depth_results(all_results, label="OVERALL")


if __name__ == "__main__":
    main()
