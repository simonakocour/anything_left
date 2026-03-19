#!/usr/bin/env python3
"""Compute depth difference masks from before/after depth images."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remove360.preprocessing.depth_diff import batch_process_depths


def main():
    parser = argparse.ArgumentParser(
        description="Compute depth difference masks (Remove360)"
    )
    parser.add_argument(
        "depth_before_dir",
        type=str,
        help="Directory of depth images before removal",
    )
    parser.add_argument(
        "depth_after_dir",
        type=str,
        help="Directory of depth images after removal",
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Output directory for masks",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=1000,
        help="GHT nu parameter",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=300,
        help="GHT tau parameter",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate difference images",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Recompute all (do not skip existing)",
    )
    args = parser.parse_args()

    outputs = batch_process_depths(
        args.depth_before_dir,
        args.depth_after_dir,
        args.output_dir,
        nu=args.nu,
        tau=args.tau,
        save_intermediate=args.save_intermediate,
        skip_existing=not args.no_skip,
    )
    print(f"Processed {len(outputs)} depth pairs -> {args.output_dir}")


if __name__ == "__main__":
    main()
