"""
temporal_aligner.py
--------------------
Aligns ERA5/NOAA weather data with Sentinel-2 satellite imagery timestamps.

WHY TEMPORAL ALIGNMENT MATTERS:
  Sentinel-2 captures images every 5 days (at best).
  Weather data is available daily or hourly.

  To train our CNN-LSTM model, each satellite image patch needs:
    → The weather conditions from the DAYS BEFORE the image was taken
    → Because fire risk depends on accumulated weather (not just one day)

  Example:
    Satellite image taken: 2023-07-15
    We need weather from:  2023-07-01 to 2023-07-15 (14-day lookback)
    This becomes the LSTM's time-series input for that image.

ALIGNMENT STRATEGY:
  For each satellite tile:
    1. Find the acquisition date from the filename or metadata
    2. Extract weather data for N days before that date (lookback window)
    3. Aggregate to daily statistics (mean, max, min)
    4. Return as a time-series array of shape (N_days, N_weather_features)

  Final model input per sample:
    - Satellite patch  : (6_bands, 64, 64)       ← CNN processes this
    - Weather series   : (14_days, 8_features)   ← LSTM processes this
    - Label            : fire_risk (0/1/2)        ← model predicts this
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


# ─── WEATHER FEATURE COLUMNS ─────────────────────────────────────────────────
# These are the features we extract for the LSTM time series
WEATHER_FEATURES = [
    "temp_max_c",           # Max temperature
    "temp_min_c",           # Min temperature
    "precipitation_mm",     # Daily precipitation
    "wind_speed_ms",        # Wind speed
    "humidity_min",         # Minimum humidity (most relevant for fire)
    "fire_danger_score",    # Composite danger score
    "consecutive_dry_days", # Accumulated dryness
    "vpd_kpa",              # Vapor pressure deficit (if from ERA5)
]


# ─── CORE ALIGNMENT FUNCTION ─────────────────────────────────────────────────

def align_weather_to_satellite(
    satellite_date: str,
    weather_df: pd.DataFrame,
    lookback_days: int = 14,
    date_column: str = "date"
) -> np.ndarray:
    """
    Extract weather time series for the days leading up to a satellite image.

    Parameters:
        satellite_date : Date when satellite image was taken (YYYY-MM-DD)
        weather_df     : DataFrame with daily weather data
        lookback_days  : How many days before the image to include (default 14)
        date_column    : Name of the date column in weather_df

    Returns:
        2D numpy array of shape (lookback_days, n_features)
        Each row = one day of weather, each column = one feature
        Ordered from oldest to most recent (day -14 to day -1)

    Example:
        satellite_date = "2023-07-15"
        lookback_days  = 14
        → Returns weather from 2023-07-01 to 2023-07-14
        → Shape: (14, 8) if using all 8 WEATHER_FEATURES
    """
    sat_date = pd.Timestamp(satellite_date)
    start_date = sat_date - pd.Timedelta(days=lookback_days)

    # Filter weather to the lookback window
    mask = (
        (weather_df[date_column] >= start_date) &
        (weather_df[date_column] < sat_date)
    )
    window = weather_df[mask].copy()
    window = window.sort_values(date_column).reset_index(drop=True)

    # Get available feature columns
    available = [f for f in WEATHER_FEATURES if f in window.columns]

    if len(window) == 0:
        print(f"[WARN] No weather data found for window ending {satellite_date}")
        return np.full((lookback_days, len(available)), np.nan)

    # Extract feature array
    feature_array = window[available].values.astype(float)

    # Pad or trim to exact lookback_days length
    feature_array = _pad_or_trim(feature_array, lookback_days)

    return feature_array


def _pad_or_trim(array: np.ndarray, target_length: int) -> np.ndarray:
    """
    Ensure array has exactly target_length rows.
    - If too short: pad beginning with NaN (missing older data)
    - If too long: keep most recent rows
    """
    current_length, n_features = array.shape

    if current_length == target_length:
        return array

    elif current_length < target_length:
        # Pad with NaN at the beginning (older dates missing)
        pad = np.full((target_length - current_length, n_features), np.nan)
        return np.vstack([pad, array])

    else:
        # Too many rows — keep most recent
        return array[-target_length:]


# ─── BATCH ALIGNMENT ─────────────────────────────────────────────────────────

def create_aligned_dataset(
    satellite_dates: list,
    weather_df: pd.DataFrame,
    satellite_patches: Optional[np.ndarray] = None,
    lookback_days: int = 14,
    fill_nan_method: str = "interpolate"
) -> dict:
    """
    Create a complete aligned dataset for all satellite acquisitions.

    This is the main function that prepares data for model training.
    It produces matched pairs of (satellite_patch, weather_series).

    Parameters:
        satellite_dates   : List of date strings when images were taken
        weather_df        : DataFrame with daily weather data
        satellite_patches : Optional array (N_dates, bands, H, W)
        lookback_days     : Days of weather history per sample
        fill_nan_method   : How to handle missing weather ('interpolate', 'zero', 'mean')

    Returns:
        Dictionary with:
          'weather_series'  : (N_dates, lookback_days, n_features)
          'satellite_dates' : List of dates
          'feature_names'   : List of weather feature names used
          'satellite_patches': Passed-through if provided
          'summary'         : Statistics about alignment quality
    """
    print(f"[INFO] Aligning {len(satellite_dates)} satellite dates with weather data...")
    print(f"       Lookback window : {lookback_days} days")
    print(f"       Weather records : {len(weather_df):,}")

    available_features = [f for f in WEATHER_FEATURES if f in weather_df.columns]
    print(f"       Weather features: {available_features}")

    weather_series = []
    valid_dates    = []
    skipped_dates  = []

    for date_str in satellite_dates:
        series = align_weather_to_satellite(
            satellite_date = date_str,
            weather_df     = weather_df,
            lookback_days  = lookback_days
        )

        # Handle NaN values
        series = _fill_nan(series, method=fill_nan_method)

        nan_frac = np.isnan(series).mean()
        if nan_frac > 0.5:
            # Skip if more than 50% is missing
            skipped_dates.append(date_str)
            continue

        weather_series.append(series)
        valid_dates.append(date_str)

    if len(weather_series) == 0:
        print("[ERROR] No valid aligned samples found!")
        return {}

    weather_array = np.stack(weather_series, axis=0)

    # Summary statistics
    total     = len(satellite_dates)
    valid     = len(valid_dates)
    skipped   = len(skipped_dates)
    nan_frac  = np.isnan(weather_array).mean()

    print(f"\n[INFO] Alignment complete:")
    print(f"       Total dates   : {total}")
    print(f"       Valid samples : {valid}")
    print(f"       Skipped       : {skipped} (>50% missing weather)")
    print(f"       NaN fraction  : {nan_frac:.3f}")
    print(f"       Output shape  : {weather_array.shape}")
    print(f"       (samples, days, features) = {weather_array.shape}")

    result = {
        "weather_series"   : weather_array,
        "satellite_dates"  : valid_dates,
        "feature_names"    : available_features,
        "n_lookback_days"  : lookback_days,
        "summary": {
            "total_dates"  : total,
            "valid_samples": valid,
            "skipped"      : skipped,
            "nan_fraction" : float(nan_frac),
        }
    }

    if satellite_patches is not None:
        result["satellite_patches"] = satellite_patches[:len(valid_dates)]

    return result


def _fill_nan(array: np.ndarray, method: str = "interpolate") -> np.ndarray:
    """Fill NaN values in weather time series."""
    if not np.any(np.isnan(array)):
        return array

    result = array.copy()

    if method == "interpolate":
        # Linear interpolation along time axis for each feature
        for col in range(array.shape[1]):
            series = pd.Series(array[:, col])
            result[:, col] = series.interpolate(
                method="linear", limit_direction="both"
            ).fillna(series.mean()).values

    elif method == "zero":
        result = np.nan_to_num(result, nan=0.0)

    elif method == "mean":
        col_means = np.nanmean(array, axis=0)
        for col in range(array.shape[1]):
            mask = np.isnan(result[:, col])
            result[mask, col] = col_means[col]

    return result


# ─── NORMALIZATION ────────────────────────────────────────────────────────────

def normalize_weather_series(
    weather_array: np.ndarray,
    method: str = "minmax"
) -> tuple:
    """
    Normalize weather time series for LSTM input.

    Parameters:
        weather_array : 3D array (N_samples, N_days, N_features)
        method        : 'minmax' or 'zscore'

    Returns:
        Tuple of (normalized_array, normalization_stats)
        Stats needed to inverse-transform predictions later.
    """
    N, T, F = weather_array.shape
    normalized = weather_array.copy().astype(float)
    stats = {}

    print(f"[INFO] Normalizing weather series: shape={weather_array.shape}")

    for f in range(F):
        feature_data = weather_array[:, :, f].flatten()
        valid = feature_data[~np.isnan(feature_data)]

        if len(valid) == 0:
            stats[f] = {"method": method, "min": 0, "max": 1, "mean": 0, "std": 1}
            continue

        if method == "minmax":
            vmin, vmax = valid.min(), valid.max()
            if vmax - vmin < 1e-8:
                normalized[:, :, f] = 0.0
            else:
                normalized[:, :, f] = (weather_array[:, :, f] - vmin) / (vmax - vmin)
            stats[f] = {"method": "minmax", "min": float(vmin), "max": float(vmax)}

        elif method == "zscore":
            mean, std = valid.mean(), valid.std()
            if std < 1e-8:
                std = 1.0
            normalized[:, :, f] = (weather_array[:, :, f] - mean) / std
            stats[f] = {"method": "zscore", "mean": float(mean), "std": float(std)}

    print(f"[INFO] Normalized range: [{np.nanmin(normalized):.3f}, {np.nanmax(normalized):.3f}]")
    return normalized, stats


# ─── SAVE / LOAD ──────────────────────────────────────────────────────────────

def save_aligned_dataset(dataset: dict, output_dir: str = "data/processed"):
    """Save aligned dataset to disk for model training."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    np.save(
        os.path.join(output_dir, "weather_series.npy"),
        dataset["weather_series"]
    )

    meta = {
        "satellite_dates" : dataset["satellite_dates"],
        "feature_names"   : dataset["feature_names"],
        "n_lookback_days" : dataset["n_lookback_days"],
        "summary"         : dataset["summary"],
    }
    import json
    with open(os.path.join(output_dir, "weather_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] Dataset saved to: {output_dir}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, "..")

    print("=== Temporal Aligner — Alaska Wildfire Pipeline ===\n")

    # Import weather generators
    from era5_fetcher import generate_synthetic_era5
    from noaa_fetcher import generate_synthetic_noaa

    # Step 1: Generate synthetic weather data (90 days, fire season)
    print("--- Step 1: Generate weather data ---")
    era5_df  = generate_synthetic_era5(region="interior", n_days=90)
    noaa_df  = generate_synthetic_noaa(
        latitude=64.815, longitude=-147.856,
        start_date="2023-06-01", end_date="2023-08-31"
    )

    # Merge ERA5 and NOAA (average where both available)
    print("\n--- Step 2: Merge ERA5 + NOAA data ---")
    era5_daily = era5_df.groupby("time").agg({
        "temp_c"        : "mean",
        "rel_humidity"  : "mean",
        "wind_speed"    : "mean",
        "precipitation_mm": "sum",
        "vpd_kpa"       : "mean",
    }).reset_index().rename(columns={
        "time"          : "date",
        "temp_c"        : "temp_max_c",
        "rel_humidity"  : "humidity_min",
        "wind_speed"    : "wind_speed_ms",
    })
    era5_daily["date"] = pd.to_datetime(era5_daily["date"])

    # Use NOAA as primary, add VPD from ERA5
    noaa_df = noaa_df.merge(
        era5_daily[["date", "vpd_kpa"]],
        on="date", how="left"
    )
    print(f"[INFO] Merged weather features: {list(noaa_df.columns)}")

    # Step 3: Simulate satellite acquisition dates
    # Sentinel-2 has ~5 day revisit in Alaska
    print("\n--- Step 3: Simulate Sentinel-2 acquisition dates ---")
    sat_dates = [
        str((pd.Timestamp("2023-06-01") + pd.Timedelta(days=i*5)).date())
        for i in range(18)  # 18 acquisitions over 90 days
    ]
    print(f"[INFO] Satellite dates: {sat_dates[:5]}... ({len(sat_dates)} total)")

    # Step 4: Align weather to satellite dates
    print("\n--- Step 4: Align weather to satellite dates ---")
    dataset = create_aligned_dataset(
        satellite_dates = sat_dates,
        weather_df      = noaa_df,
        lookback_days   = 14
    )

    # Step 5: Normalize
    print("\n--- Step 5: Normalize weather series ---")
    normalized, stats = normalize_weather_series(
        dataset["weather_series"], method="minmax"
    )

    print(f"\n=== Final Dataset Ready for CNN-LSTM ===")
    print(f"  Weather series shape : {normalized.shape}")
    print(f"  (N_samples, N_days, N_features) = {normalized.shape}")
    print(f"  Features used: {dataset['feature_names']}")
    print(f"\n  This gets fed into the LSTM part of the CNN-LSTM model.")
    print(f"  The CNN processes the satellite patches (spatial features).")
    print(f"  The LSTM processes these weather series (temporal features).")
    print(f"  Combined → fire risk prediction.")

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/weather_series_test.npy", normalized)
    print(f"\n[DONE] Saved test weather series to data/processed/weather_series_test.npy")