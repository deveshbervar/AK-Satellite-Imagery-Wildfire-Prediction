"""
normalizer.py
-------------
Normalizes Sentinel-2 bands for input into a deep learning model.

WHY NORMALIZATION MATTERS:
  Raw Sentinel-2 pixel values are surface reflectance values (0–10000).
  Neural networks train much better when inputs are in range [0, 1] or [-1, 1].
  Without normalization, the model struggles to converge.

METHODS WE SUPPORT:
  1. min_max   : Scales each band to [0, 1] using its own min/max
  2. percentile: Clips outliers (2nd–98th percentile) then scales to [0, 1]
                 → Best for satellite imagery (avoids cloud/shadow outliers)
  3. z_score   : Subtract mean, divide by std. Output is centered at 0
                 → Good for CNN input layers

TILE PATCHING:
  Satellite images are very large (e.g., 10000 x 10000 pixels).
  CNNs work on fixed-size patches (e.g., 64x64 or 128x128).
  We must split the image into overlapping patches for model training.
"""

import numpy as np
from pathlib import Path
import json


# ─── NORMALIZATION ────────────────────────────────────────────────────────────

def normalize_minmax(bands: np.ndarray) -> np.ndarray:
    """
    Scale each band independently to [0, 1] using min-max normalization.

    Parameters:
        bands : 3D array (n_bands x H x W)

    Returns:
        Normalized array, same shape, values in [0, 1]
    """
    normalized = np.zeros_like(bands, dtype=float)

    for i in range(bands.shape[0]):
        band = bands[i]
        valid = band[~np.isnan(band)]

        if len(valid) == 0:
            print(f"[WARN] Band {i} is all NaN, skipping.")
            continue

        bmin, bmax = valid.min(), valid.max()

        if bmax - bmin < 1e-8:
            print(f"[WARN] Band {i} has near-zero range ({bmin:.4f}), setting to 0.")
            normalized[i] = 0.0
        else:
            normalized[i] = (band - bmin) / (bmax - bmin)

    return normalized


def normalize_percentile(bands: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """
    Scale each band to [0, 1] using percentile clipping.

    This is RECOMMENDED for satellite imagery because:
    - Shadow pixels are very dark outliers
    - Remaining clouds (if any) are very bright outliers
    - Percentile clipping removes these extremes before scaling

    Parameters:
        bands : 3D array (n_bands x H x W)
        low   : Lower percentile to clip (default 2%)
        high  : Upper percentile to clip (default 98%)

    Returns:
        Normalized array, values in [0, 1]
    """
    normalized = np.zeros_like(bands, dtype=float)

    for i in range(bands.shape[0]):
        band = bands[i]
        valid = band[~np.isnan(band)]

        if len(valid) == 0:
            continue

        p_low  = np.percentile(valid, low)
        p_high = np.percentile(valid, high)

        clipped = np.clip(band, p_low, p_high)
        normalized[i] = (clipped - p_low) / (p_high - p_low + 1e-8)

    return normalized


def normalize_zscore(bands: np.ndarray) -> tuple:
    """
    Z-score normalization: subtract mean, divide by std.
    Output is centered around 0, not bounded to [0, 1].

    Returns:
        Tuple of (normalized_bands, stats_dict)
        stats_dict contains per-band mean and std for later inverse transform.
    """
    normalized = np.zeros_like(bands, dtype=float)
    stats = {}

    for i in range(bands.shape[0]):
        band  = bands[i]
        valid = band[~np.isnan(band)]

        if len(valid) == 0:
            stats[i] = {"mean": 0, "std": 1}
            continue

        mean = np.nanmean(band)
        std  = np.nanstd(band)

        if std < 1e-8:
            std = 1.0

        normalized[i] = (band - mean) / std
        stats[i] = {"mean": float(mean), "std": float(std)}

    return normalized, stats


# ─── TILE PATCHING ────────────────────────────────────────────────────────────

def extract_patches(bands: np.ndarray, patch_size: int = 64, stride: int = 32) -> np.ndarray:
    """
    Split a large satellite image into fixed-size patches for CNN input.

    Parameters:
        bands      : 3D array (n_bands x H x W)
        patch_size : Size of each square patch (default: 64x64 pixels)
        stride     : Step between patches. stride < patch_size = overlap.
                     Overlap helps avoid boundary artifacts in predictions.

    Returns:
        4D array of shape (n_patches, n_bands, patch_size, patch_size)

    Example:
        Image: 256x256, patch_size=64, stride=32
        → Produces 49 patches (7x7 grid with 50% overlap)
    """
    n_bands, H, W = bands.shape
    patches = []
    positions = []   # Store (row, col) to reconstruct later

    row = 0
    while row + patch_size <= H:
        col = 0
        while col + patch_size <= W:
            patch = bands[:, row:row + patch_size, col:col + patch_size]

            # Skip patches that are mostly NaN (cloudy/invalid)
            nan_fraction = np.mean(np.isnan(patch))
            if nan_fraction < 0.5:   # Keep patch if < 50% is NaN
                patches.append(patch)
                positions.append((row, col))

            col += stride
        row += stride

    if len(patches) == 0:
        print("[WARN] No valid patches extracted. Image may be too cloudy.")
        return np.array([]), []

    patch_array = np.stack(patches, axis=0)   # Shape: (N, bands, H, W)
    print(f"[INFO] Extracted {len(patches)} patches of size {patch_size}x{patch_size}")
    print(f"       Patch array shape: {patch_array.shape}")

    return patch_array, positions


def reconstruct_from_patches(patches: np.ndarray, positions: list, image_shape: tuple,
                              patch_size: int = 64) -> np.ndarray:
    """
    Reconstruct a full image from predicted patches.
    Uses averaging where patches overlap.

    Parameters:
        patches     : 3D array (n_patches, H_patch, W_patch) — model predictions
        positions   : List of (row, col) tuples from extract_patches
        image_shape : (H, W) of original image
        patch_size  : Must match what was used in extract_patches

    Returns:
        2D array (H, W) — reconstructed prediction map
    """
    H, W = image_shape
    output = np.zeros((H, W), dtype=float)
    count  = np.zeros((H, W), dtype=float)   # Track how many patches cover each pixel

    for patch, (row, col) in zip(patches, positions):
        output[row:row + patch_size, col:col + patch_size] += patch
        count[row:row + patch_size,  col:col + patch_size] += 1

    # Average overlapping regions
    count[count == 0] = 1   # Avoid division by zero
    return output / count


# ─── PIPELINE: ALL STEPS TOGETHER ────────────────────────────────────────────

def prepare_for_model(
    bands: np.ndarray,
    method: str = "percentile",
    patch_size: int = 64,
    stride: int = 32
) -> tuple:
    """
    Full preprocessing pipeline:
      1. Normalize bands
      2. Extract patches
      3. Return ready-to-use model input

    Parameters:
        bands      : 3D array (n_bands x H x W), cloud-masked
        method     : Normalization method ('minmax', 'percentile', 'zscore')
        patch_size : Size of patches for CNN
        stride     : Overlap between patches

    Returns:
        Tuple of (patches, positions, norm_stats)
    """
    print(f"[INFO] Preparing data for model...")
    print(f"       Method    : {method}")
    print(f"       Patch size: {patch_size}x{patch_size}")
    print(f"       Stride    : {stride}")

    # Step 1: Normalize
    if method == "minmax":
        norm_bands = normalize_minmax(bands)
        norm_stats = {}
    elif method == "percentile":
        norm_bands = normalize_percentile(bands)
        norm_stats = {}
    elif method == "zscore":
        norm_bands, norm_stats = normalize_zscore(bands)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: minmax, percentile, zscore")

    print(f"[INFO] Normalization complete. Value range: [{np.nanmin(norm_bands):.3f}, {np.nanmax(norm_bands):.3f}]")

    # Step 2: Extract patches
    patches, positions = extract_patches(norm_bands, patch_size=patch_size, stride=stride)

    return patches, positions, norm_stats


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Normalizer Test ===")

    # Create fake satellite bands (6 bands, 256x256)
    np.random.seed(42)
    fake_bands = np.random.uniform(0, 10000, (6, 256, 256))

    # Add some NaN pixels (simulating cloud mask)
    fake_bands[:, 50:80, 50:80] = np.nan

    print("\n--- Testing percentile normalization ---")
    patches, positions, stats = prepare_for_model(fake_bands, method="percentile", patch_size=64, stride=32)

    if len(patches) > 0:
        print(f"\nReady for model:")
        print(f"  Patches shape : {patches.shape}")
        print(f"  Value range   : [{patches.min():.3f}, {patches.max():.3f}]")
        print(f"  NaN fraction  : {np.mean(np.isnan(patches)):.3f}")