"""I/O utilities for masks and JSON results."""

import json
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image


def load_mask(path: str) -> Optional[np.ndarray]:
    """Load a grayscale mask from disk.

    Args:
        path: Path to the mask image.

    Returns:
        Mask as numpy array (uint8) or None if file does not exist.
    """
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def load_binary_mask(image_path: str, threshold: int = 128) -> np.ndarray:
    """Load and binarize a mask image.

    Args:
        image_path: Path to the mask image.
        threshold: Pixel value threshold for binarization (default: 128).

    Returns:
        Binary mask as uint8 array (0 or 1).
    """
    image = Image.open(image_path).convert("L")
    mask = np.array(image)
    return (mask > threshold).astype(np.uint8)


def load_sam_data(dir_path: str) -> dict:
    """Load SAM segmentation data from a directory.

    Supports:
    - merged.json (legacy): single file with {image_name: [segments]}
    - Per-image: {image_name}.json files, each with [segments]

    Returns:
        Dict mapping image_name -> list of segment dicts.
    """
    path = Path(dir_path)
    merged = path / "merged.json"
    if merged.exists():
        with open(merged) as f:
            return json.load(f)
    result = {}
    for jf in path.glob("*.json"):
        with open(jf) as f:
            result[jf.stem] = json.load(f)
    return result


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to a JSON file.

    Args:
        data: Serializable data to save.
        path: Output file path.
        indent: JSON indentation level.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
