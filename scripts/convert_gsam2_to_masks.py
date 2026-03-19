#!/usr/bin/env python3
"""Convert Grounded-SAM-2 JSON output to PNG masks for Remove360 evaluation.

Grounded-SAM-2 saves annotations with RLE-encoded segmentation. This script
decodes them and writes one binary mask per image to the expected layout.

Usage:
  python scripts/convert_gsam2_to_masks.py outputs/ scene/gsam2/after/mask/ --prompt "table"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def decode_rle(rle: dict) -> "np.ndarray":
    """Decode RLE mask to binary numpy array."""
    try:
        from pycocotools import mask as mask_utils
        import numpy as np
    except ImportError:
        raise ImportError("Install pycocotools: pip install pycocotools")

    if isinstance(rle.get("counts"), list):
        # Uncompressed RLE
        rle = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])
    decoded = mask_utils.decode(rle)
    return (decoded > 0).astype("uint8")


def convert_gsam2_output(
    input_dir: str,
    output_dir: str,
    prompt: str = "",
    merge_instances: bool = True,
    skip_existing: bool = True,
) -> int:
    """Convert Grounded-SAM-2 JSON outputs to PNG masks.

    Args:
        input_dir: Directory containing JSON files from Grounded-SAM-2
                   (DUMP_JSON_RESULTS=True). Each file should have annotations
                   with segmentation (RLE).
        output_dir: Output directory for PNG masks (e.g. gsam2/after/mask/)
        prompt: Optional class name to filter (matches class_name). If empty, use all.
        merge_instances: If True, merge all matching instances into one mask per image.

    Returns:
        Number of images processed.
    """
    import numpy as np
    from PIL import Image

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for json_file in sorted(in_path.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        if not annotations:
            continue

        # Filter by prompt if given
        if prompt:
            annotations = [a for a in annotations if a.get("class_name", "").lower() == prompt.lower()]

        if not annotations:
            continue

        # Decode and merge masks
        h, w = data.get("img_height", 0), data.get("img_width", 0)
        if (h == 0 or w == 0) and annotations:
            seg = annotations[0].get("segmentation", {})
            if isinstance(seg, dict) and "size" in seg:
                h, w = seg["size"][0], seg["size"][1]

        merged = np.zeros((h, w), dtype=np.uint8)
        for ann in annotations:
            seg = ann.get("segmentation")
            if not seg:
                continue
            try:
                mask = decode_rle(seg)
                if mask.shape[:2] != (h, w):
                    from PIL import Image as PILImage
                    mask = np.array(
                        PILImage.fromarray(mask).resize((w, h), resample=PILImage.NEAREST)
                    )
                merged = np.maximum(merged, mask)
            except Exception:
                continue

        if merged.max() == 0:
            continue

        # Output filename from image_path or JSON filename
        img_path = data.get("image_path", str(json_file))
        stem = Path(img_path).stem or json_file.stem
        out_file = out_path / f"{stem}.png"
        if skip_existing and out_file.exists():
            continue
        Image.fromarray((merged * 255).astype(np.uint8)).save(out_file)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert Grounded-SAM-2 JSON output to PNG masks for Remove360"
    )
    parser.add_argument("input_dir", help="Directory with Grounded-SAM-2 JSON output")
    parser.add_argument("output_dir", help="Output directory for PNG masks")
    parser.add_argument("--prompt", default="", help="Filter by class name (e.g. 'table')")
    parser.add_argument("--no-skip", action="store_true", help="Recompute all (do not skip existing)")
    args = parser.parse_args()

    try:
        n = convert_gsam2_output(
            args.input_dir, args.output_dir, args.prompt,
            skip_existing=not args.no_skip,
        )
        print(f"Converted {n} images to {args.output_dir}")
    except ImportError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
