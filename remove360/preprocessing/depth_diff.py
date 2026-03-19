"""Depth difference computation using Generalized Histogram Thresholding (GHT)."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def _preliminaries(
    n: np.ndarray,
    x: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, np.ndarray, np.ndarray]:
    """Compute preliminary statistics for GHT."""
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    clip = lambda z: np.maximum(1e-30, z)
    csum = lambda z: np.cumsum(z)[:-1]
    dsum = lambda z: np.cumsum(z[::-1])[-2::-1]

    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x**2) - w0 * mu0**2
    d1 = dsum(n * x**2) - w1 * mu1**2
    return x, w0, w1, p0, p1, mu0, mu1, d0, d1


def _ght(
    n: np.ndarray,
    x: Optional[np.ndarray] = None,
    nu: float = 0,
    tau: float = 0,
    kappa: float = 0,
    omega: float = 0.5,
    prelim: Optional[Tuple] = None,
) -> Tuple[float, float]:
    """Generalized Histogram Thresholding.

    Args:
        n: Histogram counts.
        x: Bin centers (default: 0..len(n)-1).
        nu, tau, kappa, omega: GHT parameters.

    Returns:
        (threshold, objective_value)
    """
    assert nu >= 0 and tau >= 0 and kappa >= 0 and 0 <= omega <= 1
    prelim = prelim or _preliminaries(n, x)
    x, w0, w1, p0, p1, _, _, d0, d1 = prelim
    clip = lambda z: np.maximum(1e-30, z)
    argmax = lambda x_arr, f: np.mean(x_arr[:-1][f == np.max(f)])

    v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa * omega) * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
    threshold = argmax(x, f0 + f1)
    obj = f0 + f1
    return float(threshold), float(np.max(obj))


def _im2hist(im: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert image to histogram and bin centers."""
    max_val = np.iinfo(im.dtype).max
    x = np.arange(max_val + 1)
    edges = np.arange(-0.5, max_val + 1.5)
    im_bw = np.amax(im[..., :3], axis=-1) if im.ndim == 3 else im
    n, _ = np.histogram(im_bw.ravel(), edges)
    return n, x, im_bw


def process_depth_pair(
    depth_before_path: str,
    depth_after_path: str,
    save_dir: str,
    nu: float = 1000,
    tau: float = 300,
    save_intermediate: bool = True,
) -> str:
    """Compute depth difference mask for a single before/after pair.

    Uses GHT to threshold the absolute depth difference and produce a binary
    mask indicating regions of change.

    Args:
        depth_before_path: Path to depth image before removal.
        depth_after_path: Path to depth image after removal.
        save_dir: Directory to save output mask.
        nu, tau: GHT parameters (defaults from original implementation).
        save_intermediate: Whether to save intermediate difference image.

    Returns:
        Path to the output mask file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    depth1 = np.array(Image.open(depth_before_path).convert("L"))
    depth2 = np.array(Image.open(depth_after_path).convert("L"))

    diff = depth2.astype(np.float32) - depth1.astype(np.float32)
    diff[(diff == depth2) | (diff == -depth1)] = 0
    diff_abs = np.abs(diff)

    if diff_abs.max() < 1e-6:
        import warnings
        warnings.warn(
            f"Depth diff is zero for {depth_before_path} vs {depth_after_path}. "
            "Before and after are identical—check that you're using different depth dirs.",
            UserWarning,
            stacklevel=2,
        )

    prefix = Path(depth_before_path).stem
    interm_path = Path(save_dir) / f"{prefix}_interm.png"

    if save_intermediate:
        plt.imsave(str(interm_path), diff_abs, cmap=plt.cm.gray_r)
        im = np.array(Image.open(interm_path))
    else:
        im = (np.clip(diff_abs, 0, 255)).astype(np.uint8)

    n, x, im_bw = _im2hist(im)
    prelim = _preliminaries(n, x)
    t, _ = _ght(n, x, nu=nu, tau=tau, kappa=0.0, omega=0.0, prelim=prelim)
    mask = im_bw < t

    out_path = Path(save_dir) / f"{prefix}_final_threshold.png"
    plt.imsave(str(out_path), mask, cmap=plt.cm.gray_r)
    return str(out_path)


def batch_process_depths(
    depth_before_dir: str,
    depth_after_dir: str,
    save_dir: str,
    nu: float = 1000,
    tau: float = 300,
    save_intermediate: bool = False,
    skip_existing: bool = True,
) -> list:
    """Process all matching depth pairs in two directories.

    Args:
        depth_before_dir: Directory of depth images before removal.
        depth_after_dir: Directory of depth images after removal.
        save_dir: Output directory for masks.
        nu, tau: GHT parameters.
        save_intermediate: Save intermediate difference images.
        skip_existing: Skip pairs where output mask already exists.

    Returns:
        List of output mask paths.
    """
    before_path = Path(depth_before_dir)
    after_path = Path(depth_after_dir)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    before_files = sorted(f for f in before_path.iterdir() if f.suffix.lower() == ".png")
    after_files = {f.name: f for f in after_path.iterdir() if f.suffix.lower() == ".png"}

    outputs = []
    for bf in before_files:
        if bf.name not in after_files:
            continue
        out_mask = save_path / f"{bf.stem}_final_threshold.png"
        if skip_existing and out_mask.exists():
            outputs.append(str(out_mask))
            continue
        af = after_files[bf.name]
        out = process_depth_pair(
            str(bf),
            str(af),
            save_dir,
            nu=nu,
            tau=tau,
            save_intermediate=save_intermediate,
        )
        outputs.append(out)
    return outputs
