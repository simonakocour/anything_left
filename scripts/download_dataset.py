#!/usr/bin/env python3
"""Download Remove360 dataset from Hugging Face into evaluation-ready layout.

After download, each scene has:
    {output_dir}/{scene}_{object}/
        masks/             <- GT object masks (same name as HF)
        images/before/     <- GT before-removal images (from HF train/)
        images/after/      <- GT after-removal images (from HF test/)

You then add your method renders (rgb/before/, rgb/after/, depth/, etc.) and run evaluation.
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: Install huggingface_hub: pip install huggingface_hub")
    raise SystemExit(1)


REPO_ID = "simkoc/Remove360"
DEFAULT_OUTPUT_DIR = "./data/remove360"


def main():
    parser = argparse.ArgumentParser(
        description="Download Remove360 dataset from Hugging Face (evaluation layout)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face cache directory",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking (default: symlink)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID}...")
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=out_dir / "_repo",
        cache_dir=args.cache_dir,
    )
    repo_path = Path(local_dir)

    # HF layout: scene/object/masks/, scene/object/train/, scene/object/test/
    # We create: scene_object/masks/, scene_object/images/before/, scene_object/images/after/
    scene_count = 0

    for scene_dir in sorted(repo_path.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith("."):
            continue
        for object_dir in sorted(scene_dir.iterdir()):
            if not object_dir.is_dir():
                continue
            masks_src = object_dir / "masks"
            train_src = object_dir / "train"
            test_src = object_dir / "test"
            if not masks_src.exists():
                continue

            scene_name = f"{scene_dir.name}_{object_dir.name}"
            scene_out = out_dir / scene_name
            gt_mask_out = scene_out / "masks"
            images_before_out = scene_out / "images" / "before"
            images_after_out = scene_out / "images" / "after"

            gt_mask_out.mkdir(parents=True, exist_ok=True)
            _link_contents(masks_src, gt_mask_out, args.copy)
            if train_src.exists():
                images_before_out.mkdir(parents=True, exist_ok=True)
                _link_contents(train_src, images_before_out, args.copy)
            if test_src.exists():
                images_after_out.mkdir(parents=True, exist_ok=True)
                _link_contents(test_src, images_after_out, args.copy)

            scene_count += 1
            parts = ["masks", "images/before", "images/after"]
            print(f"  {scene_name} -> {', '.join(parts)}")

    # If we copied, remove raw repo to save space. If symlinked, keep _repo.
    if args.copy:
        shutil.rmtree(repo_path, ignore_errors=True)

    print(f"\nDone. {scene_count} scenes in {out_dir}")
    print("Example: python scripts/run_pipeline.py", str(out_dir / "bedroom_table"))


def _link_contents(src: Path, dst: Path, copy: bool) -> None:
    """Link or copy all files from src to dst."""
    for f in src.iterdir():
        if f.is_file():
            target = dst / f.name
            if target.exists():
                continue
            if copy:
                shutil.copy2(f, target)
            else:
                target.symlink_to(f.resolve())


if __name__ == "__main__":
    main()
