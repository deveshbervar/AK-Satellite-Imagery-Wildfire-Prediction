"""
band_extractor.py
-----------------
Extracts and computes spectral bands and vegetation indices from Sentinel-2 imagery.

SENTINEL-2 BANDS WE USE:
  Band  | Name  | Wavelength | Resolution | Why We Use It
  ------|-------|------------|------------|-----------------------------
  B2    | Blue  | 490nm      | 10m        | RGB visualization
  B3    | Green | 560nm      | 10m        | RGB visualization
  B4    | Red   | 665nm      | 10m        | RGB + NDVI calculation
  B8    | NIR   | 842nm      | 10m        | NDVI, vegetation health
  B11   | SWIR1 | 1610nm     | 20m        | Fuel moisture (NBR index)
  B12   | SWIR2 | 2190nm     | 20m        | Burn severity (NBR index)

VEGETATION INDICES:
  1. NDVI  = (NIR - Red) / (NIR + Red)
     → Measures vegetation health. Range: -1 to +1
     → High NDVI (>0.5) = dense green vegetation = potential fuel
     → Low NDVI (<0.2) = bare soil / water / burned area

  2. NBR   = (NIR - SWIR2) / (NIR + SWIR2)
     → Normalized Burn Ratio — detects burned areas
     → Pre-fire high NBR, Post-fire low NBR → dNBR = burn severity

  3. NDMI  = (NIR - SWIR1) / (NIR + SWIR1)
     → Normalized Difference Moisture Index
     → Detects fuel moisture — dry fuel = HIGH fire risk
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ─── BAND INDEX MAPPING ───────────────────────────────────────────────────────
# Matches the band order from downloader.py: B2, B3, B4, B8, B11, B12, QA60
BAND_INDEX = {
    "B2_blue"  : 0,
    "B3_green" : 1,
    "B4_red"   : 2,
    "B8_nir"   : 3,
    "B11_swir1": 4,
    "B12_swir2": 5,
}


# ─── BAND EXTRACTION ─────────────────────────────────────────────────────────

def extract_bands(masked_bands: np.ndarray) -> dict:
    """
    Extract individual named bands from stacked array.

    Parameters:
        masked_bands : 3D array (6 x H x W) — already cloud-masked

    Returns:
        Dictionary mapping band names to 2D arrays

    Usage:
        bands = extract_bands(masked_bands)
        nir = bands["B8_nir"]
        red = bands["B4_red"]
    """
    extracted = {}
    for name, idx in BAND_INDEX.items():
        if idx < masked_bands.shape[0]:
            extracted[name] = masked_bands[idx].astype(float)
        else:
            print(f"[WARN] Band {name} not found at index {idx}")

    print(f"[INFO] Extracted {len(extracted)} bands: {list(extracted.keys())}")
    return extracted


# ─── VEGETATION INDICES ───────────────────────────────────────────────────────

def compute_ndvi(bands: dict) -> np.ndarray:
    """
    NDVI = (NIR - Red) / (NIR + Red)

    Interpretation for wildfire:
      > 0.6  : Dense forest (high fuel load)
      0.2-0.6: Shrubs/grassland (moderate fuel)
      < 0.2  : Bare soil, water, snow (low risk)
      < 0.0  : Water bodies
    """
    nir = bands["B8_nir"]
    red = bands["B4_red"]

    # Avoid division by zero using np.where
    denominator = nir + red
    ndvi = np.where(denominator != 0, (nir - red) / denominator, np.nan)

    # Clip to valid range [-1, 1]
    ndvi = np.clip(ndvi, -1, 1)

    valid = np.sum(~np.isnan(ndvi))
    print(f"[INFO] NDVI computed — valid pixels: {valid:,}")
    print(f"       Mean NDVI: {np.nanmean(ndvi):.3f}")
    print(f"       Max NDVI : {np.nanmax(ndvi):.3f}")

    return ndvi


def compute_nbr(bands: dict) -> np.ndarray:
    """
    NBR = (NIR - SWIR2) / (NIR + SWIR2)

    Used to detect burned areas:
      High NBR (>0.1) : Healthy vegetation
      Low NBR (<-0.1) : Burned area

    dNBR = pre_fire_NBR - post_fire_NBR
      > 0.66 : High severity burn
      0.27-0.66: Moderate severity
      < 0.1  : Unburned
    """
    nir   = bands["B8_nir"]
    swir2 = bands["B12_swir2"]

    denominator = nir + swir2
    nbr = np.where(denominator != 0, (nir - swir2) / denominator, np.nan)
    nbr = np.clip(nbr, -1, 1)

    print(f"[INFO] NBR computed — Mean: {np.nanmean(nbr):.3f}")
    return nbr


def compute_ndmi(bands: dict) -> np.ndarray:
    """
    NDMI = (NIR - SWIR1) / (NIR + SWIR1)

    Moisture index — key for fire risk:
      High NDMI (>0.0) : Moist vegetation (lower fire risk)
      Low NDMI (<-0.2) : Dry vegetation (HIGHER fire risk)
    """
    nir   = bands["B8_nir"]
    swir1 = bands["B11_swir1"]

    denominator = nir + swir1
    ndmi = np.where(denominator != 0, (nir - swir1) / denominator, np.nan)
    ndmi = np.clip(ndmi, -1, 1)

    print(f"[INFO] NDMI computed — Mean (moisture): {np.nanmean(ndmi):.3f}")
    return ndmi


def compute_all_indices(bands: dict) -> dict:
    """Compute all vegetation indices at once."""
    print("[INFO] Computing vegetation indices...")
    return {
        "ndvi": compute_ndvi(bands),
        "nbr" : compute_nbr(bands),
        "ndmi": compute_ndmi(bands),
    }


# ─── FIRE RISK SCORING ────────────────────────────────────────────────────────

def compute_fire_risk_score(indices: dict) -> np.ndarray:
    """
    Simple rule-based fire risk score combining vegetation indices.

    Logic:
      - High NDVI = lots of dry fuel
      - Low NDMI  = fuel is dry
      - Combined → fire risk

    Risk levels:
      > 0.6 : HIGH risk
      0.3-0.6: MODERATE risk
      < 0.3 : LOW risk

    Note: This is a simple heuristic — the ML model will learn better weights.
    """
    ndvi = indices["ndvi"]
    ndmi = indices["ndmi"]

    # Normalize to [0, 1]
    ndvi_norm = (ndvi + 1) / 2      # Maps -1..+1 → 0..1
    dryness   = 1 - (ndmi + 1) / 2  # Invert moisture: dry = high score

    # Combined score: more vegetation AND more dryness = higher risk
    risk = (ndvi_norm * 0.5) + (dryness * 0.5)
    risk = np.clip(risk, 0, 1)

    # Summary
    high_risk = np.sum(risk > 0.6)
    mod_risk  = np.sum((risk > 0.3) & (risk <= 0.6))
    low_risk  = np.sum(risk <= 0.3)
    total     = np.sum(~np.isnan(risk))

    print(f"\n[INFO] Fire Risk Summary:")
    print(f"       HIGH     : {high_risk:,} px ({100*high_risk/total:.1f}%)")
    print(f"       MODERATE : {mod_risk:,} px ({100*mod_risk/total:.1f}%)")
    print(f"       LOW      : {low_risk:,} px ({100*low_risk/total:.1f}%)")

    return risk


# ─── SAVE INDICES ─────────────────────────────────────────────────────────────

def save_indices_to_tif(indices: dict, reference_tif: str, output_dir: str = "data/processed"):
    """
    Save computed indices as separate GeoTIFF files.
    Uses the original file's CRS and transform (geolocation info).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_tif) as src:
        meta = src.meta.copy()

    meta.update({"count": 1, "dtype": "float32", "nodata": np.nan})

    for name, data in indices.items():
        out_path = os.path.join(output_dir, f"{name}.tif")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data.astype("float32"), 1)
        print(f"[INFO] Saved {name} → {out_path}")


# ─── VISUALIZATION ────────────────────────────────────────────────────────────

def visualize_indices(bands: dict, indices: dict, risk: np.ndarray, save_path: str = None):
    """Plot RGB, NDVI, NDMI, and Fire Risk side by side."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Sentinel-2 Band Extraction & Fire Risk Analysis — Alaska", fontsize=14, fontweight="bold")

    def normalize(arr):
        arr = np.nan_to_num(arr, nan=0)
        p2, p98 = np.nanpercentile(arr[arr > 0], 2), np.nanpercentile(arr[arr > 0], 98)
        return np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)

    # ── Row 1
    # RGB
    rgb = np.dstack([normalize(bands["B4_red"]), normalize(bands["B3_green"]), normalize(bands["B2_blue"])])
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("True Color RGB\n(B4-B3-B2)")
    axes[0, 0].axis("off")

    # False color (vegetation shows as red)
    false_color = np.dstack([normalize(bands["B8_nir"]), normalize(bands["B4_red"]), normalize(bands["B3_green"])])
    axes[0, 1].imshow(false_color)
    axes[0, 1].set_title("False Color\n(NIR-R-G) — Vegetation = Red")
    axes[0, 1].axis("off")

    # NDVI
    im = axes[0, 2].imshow(indices["ndvi"], cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[0, 2].set_title("NDVI\n(Green = Dense Vegetation)")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # ── Row 2
    # NBR
    im2 = axes[1, 0].imshow(indices["nbr"], cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title("NBR\n(Red = Burned Area)")
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # NDMI
    im3 = axes[1, 1].imshow(indices["ndmi"], cmap="RdYlBu", vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title("NDMI (Moisture)\n(Blue = Moist, Red = Dry)")
    axes[1, 1].axis("off")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    # Fire Risk
    risk_cmap = mcolors.LinearSegmentedColormap.from_list("risk", ["green", "yellow", "red"])
    im4 = axes[1, 2].imshow(risk, cmap=risk_cmap, vmin=0, vmax=1)
    axes[1, 2].set_title("Fire Risk Score\n(Red = High Risk)")
    axes[1, 2].axis("off")
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved visualization to: {save_path}")
    else:
        plt.show()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys

    print("=== Band Extractor — Alaska Wildfire Pipeline ===")
    print("This module is designed to be imported, not run directly.")
    print("See notebooks/sentinel2_demo.ipynb for a full walkthrough.")
    print("\nQuick test with synthetic data:")

    # Create fake data for testing without real satellite files
    H, W = 256, 256
    np.random.seed(42)
    fake_bands = {
        "B2_blue"  : np.random.uniform(0.02, 0.1, (H, W)),
        "B3_green" : np.random.uniform(0.03, 0.12, (H, W)),
        "B4_red"   : np.random.uniform(0.02, 0.15, (H, W)),
        "B8_nir"   : np.random.uniform(0.1, 0.5, (H, W)),
        "B11_swir1": np.random.uniform(0.05, 0.3, (H, W)),
        "B12_swir2": np.random.uniform(0.02, 0.2, (H, W)),
    }

    indices = compute_all_indices(fake_bands)
    risk    = compute_fire_risk_score(indices)
    visualize_indices(fake_bands, indices, risk, save_path="band_extraction_result.png")
    print("\n[DONE] Test complete. Check 'band_extraction_result.png'")