"""
cloud_mask.py
-------------
Removes clouds from Sentinel-2 imagery using the QA60 band.

WHY THIS MATTERS:
  Clouds block the satellite's view of the ground. If we train a model
  on cloudy pixels, it learns noise instead of actual vegetation/fire signals.
  We must remove cloud pixels BEFORE any analysis.

HOW QA60 WORKS:
  Sentinel-2 includes a QA60 band — a bitmask where:
    Bit 10 = 1 → Opaque cloud (definitely cloudy)
    Bit 11 = 1 → Cirrus cloud (thin/wispy cloud)

  We check these bits and mask any pixel where either is set to 1.

  Example:
    QA60 value = 1024 = binary 10000000000 → Bit 10 is set → CLOUD
    QA60 value = 2048 = binary 100000000000 → Bit 11 is set → CIRRUS
    QA60 value = 0    = binary 00000000000 → No cloud → KEEP THIS PIXEL
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
import matplotlib.pyplot as plt


# ─── CONSTANTS ───────────────────────────────────────────────────────────────

# These are the bit positions in QA60 that indicate clouds
CLOUD_BIT    = 1 << 10   # = 1024  (opaque clouds)
CIRRUS_BIT   = 1 << 11   # = 2048  (cirrus clouds)


# ─── CORE FUNCTION ───────────────────────────────────────────────────────────

def create_cloud_mask(qa60_band: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask from the QA60 band.

    Parameters:
        qa60_band : 2D numpy array of QA60 values (shape: H x W)

    Returns:
        2D boolean array (shape: H x W)
            True  = pixel is CLEAR (keep it)
            False = pixel is CLOUDY (remove it)

    Example:
        qa60 = np.array([[0, 1024], [2048, 0]])
        mask = create_cloud_mask(qa60)
        # mask = [[True, False], [False, True]]
    """
    # Use bitwise AND to check if cloud bit is set
    cloud_mask  = (qa60_band & CLOUD_BIT)  == 0   # True where NOT cloudy
    cirrus_mask = (qa60_band & CIRRUS_BIT) == 0   # True where NOT cirrus

    # Both must be clear for a pixel to be usable
    clear_mask = cloud_mask & cirrus_mask

    return clear_mask


def apply_cloud_mask(image_bands: np.ndarray, clear_mask: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """
    Apply cloud mask to all image bands.

    Parameters:
        image_bands : 3D numpy array (shape: Bands x H x W)
        clear_mask  : 2D boolean array (shape: H x W), True = clear pixel
        fill_value  : Value to fill cloudy pixels with (default: NaN)

    Returns:
        3D numpy array with cloudy pixels replaced by fill_value
    """
    masked = image_bands.astype(float).copy()

    # Apply mask to each band
    for i in range(masked.shape[0]):
        masked[i][~clear_mask] = fill_value   # Set cloudy pixels to NaN

    return masked


# ─── FILE-LEVEL FUNCTION ─────────────────────────────────────────────────────

def mask_clouds_in_file(input_tif: str, output_tif: str = None, qa60_band_index: int = 6) -> np.ndarray:
    """
    Read a GeoTIFF, apply cloud masking, and optionally save result.

    Parameters:
        input_tif       : Path to input GeoTIFF (must include QA60 band)
        output_tif      : Path to save masked output (None = don't save)
        qa60_band_index : Which band index is QA60 (0-based, default=6 for B2,B3,B4,B8,B11,B12,QA60)

    Returns:
        Tuple of (masked_bands, clear_mask, metadata)

    Band order assumed (from downloader.py):
        Index 0 = B2  (Blue)
        Index 1 = B3  (Green)
        Index 2 = B4  (Red)
        Index 3 = B8  (NIR)
        Index 4 = B11 (SWIR1)
        Index 5 = B12 (SWIR2)
        Index 6 = QA60 (Cloud mask)
    """
    print(f"[INFO] Reading: {input_tif}")

    with rasterio.open(input_tif) as src:
        # Read all bands at once → shape: (n_bands, height, width)
        bands = src.read()
        meta  = src.meta.copy()
        print(f"       Image shape : {bands.shape}")
        print(f"       CRS         : {src.crs}")
        print(f"       Resolution  : {src.res}")

    # Extract QA60 band (0-based index)
    qa60 = bands[qa60_band_index]

    # Create cloud mask
    clear_mask = create_cloud_mask(qa60)

    # Calculate cloud coverage percentage
    total_pixels = clear_mask.size
    clear_pixels = np.sum(clear_mask)
    cloud_pct = (1 - clear_pixels / total_pixels) * 100
    print(f"[INFO] Cloud coverage: {cloud_pct:.1f}%")
    print(f"       Clear pixels  : {clear_pixels:,} / {total_pixels:,}")

    # Apply mask to spectral bands only (exclude QA60 band)
    spectral_bands = bands[:qa60_band_index]   # All bands except QA60
    masked_bands   = apply_cloud_mask(spectral_bands, clear_mask)

    # Optionally save to file
    if output_tif:
        save_masked_image(masked_bands, meta, output_tif, n_bands=qa60_band_index)
        print(f"[DONE] Saved masked image to: {output_tif}")

    return masked_bands, clear_mask, meta


def save_masked_image(bands: np.ndarray, meta: dict, output_path: str, n_bands: int):
    """Save cloud-masked bands as a new GeoTIFF."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    meta.update({
        "count": n_bands,
        "dtype": "float32",
        "nodata": np.nan
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        for i in range(n_bands):
            dst.write(bands[i].astype("float32"), i + 1)   # rasterio is 1-indexed


# ─── VISUALIZATION ───────────────────────────────────────────────────────────

def visualize_cloud_mask(bands: np.ndarray, clear_mask: np.ndarray, save_path: str = None):
    """
    Plot a side-by-side comparison of original vs cloud-masked image.

    Parameters:
        bands      : 3D array (Bands x H x W), index 2=Red, 1=Green, 0=Blue
        clear_mask : 2D boolean array
        save_path  : If given, saves the plot as PNG
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Cloud Masking Results", fontsize=14, fontweight="bold")

    # ── RGB composite (bands: R=B4=idx2, G=B3=idx1, B=B2=idx0)
    def normalize(arr):
        arr = np.nan_to_num(arr, nan=0)
        vmin, vmax = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
        return np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)

    rgb_original = np.dstack([normalize(bands[2]), normalize(bands[1]), normalize(bands[0])])
    axes[0].imshow(rgb_original)
    axes[0].set_title("Original RGB")
    axes[0].axis("off")

    # ── Cloud mask
    axes[1].imshow(clear_mask, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_title("Cloud Mask\n(Green=Clear, Red=Cloud)")
    axes[1].axis("off")

    # ── Masked result
    masked_rgb = rgb_original.copy()
    masked_rgb[~clear_mask] = [0.5, 0.5, 0.5]   # Grey out cloudy pixels
    axes[2].imshow(masked_rgb)
    axes[2].set_title("After Cloud Masking\n(Grey = removed clouds)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved visualization to: {save_path}")
    else:
        plt.show()


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cloud_mask.py <path_to_sentinel2.tif>")
        print("Example: python cloud_mask.py data/raw/sentinel2/alaska_interior.tif")
    else:
        input_file  = sys.argv[1]
        output_file = input_file.replace(".tif", "_cloudmasked.tif")

        bands, mask, meta = mask_clouds_in_file(input_file, output_tif=output_file)
        visualize_cloud_mask(bands, mask, save_path="cloud_mask_result.png")