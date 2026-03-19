#!/usr/bin/env python3
"""Evaluate semantic (GSAM) residuals for Remove360 scenes."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remove360.evaluation.semantic import evaluate_semantic_scene, summarize_semantic_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic (GSAM) residuals after object removal (Remove360)"
    )
    parser.add_argument(
        "scene_dirs",
        nargs="+",
        type=str,
        help="Paths to scene directories",
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
            results = evaluate_semantic_scene(str(path), verbose=not args.quiet)
            all_results.extend(results)
            summarize_semantic_results(results, label=path.name)
        except FileNotFoundError as e:
            print(f"Error: {e}")

    if all_results:
        summarize_semantic_results(all_results, label="OVERALL")


if __name__ == "__main__":
    main()
