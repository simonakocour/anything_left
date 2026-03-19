#!/usr/bin/env python3
"""Run SAM on RGB images to produce masks for evaluation.

Saves one {image_stem}.json per image (avoids large merged file, enables resume).
Requires: pip install segment-anything
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Paths — edit if needed
DEFAULT_SAM_CHECKPOINT = str(ROOT / "checkpoints" / "sam_vit_h_4b8939.pth")


def run_sam_on_directory(
    rgb_dir: str,
    output_dir: str,
    checkpoint: str = "sam_vit_h_4b8939.pth",
    device: str = "cuda",
    skip_existing: bool = True,
) -> str:
    """Run SAM automatic mask generator on all images in rgb_dir."""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        import numpy as np
        from PIL import Image
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError(
            "Install segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from e

    rgb_path = Path(rgb_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    images = sorted(rgb_path.glob("*.png")) + sorted(rgb_path.glob("*.jpg"))
    to_process = images
    if skip_existing:
        to_process = [p for p in images if not (out_path / f"{p.stem}.json").exists()]

    if not to_process:
        print(f"Skipped (all {len(images)} images already in {out_path})")
        return str(out_path)

    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    for img_path in tqdm(to_process, desc="SAM"):
        img = np.array(Image.open(img_path).convert("RGB"))
        masks = mask_generator.generate(img)
        segments = [
            {
                "segmentation": m["segmentation"].tolist(),
                "area": int(m["area"]),
                "bbox": m["bbox"],
            }
            for m in masks
        ]
        out_file = out_path / f"{img_path.stem}.json"
        with open(out_file, "w") as f:
            json.dump(segments, f)

    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Run SAM on RGB images")
    parser.add_argument("rgb_dir", help="Directory of RGB images")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument("--checkpoint", default=DEFAULT_SAM_CHECKPOINT, help="SAM checkpoint path")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--no-skip", action="store_true", help="Recompute all (do not skip existing)")
    args = parser.parse_args()

    try:
        out = run_sam_on_directory(
            args.rgb_dir, args.output_dir, args.checkpoint, args.device,
            skip_existing=not args.no_skip,
        )
        print(f"Saved: {out}")
    except ImportError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
