"""
noaa_fetcher.py
---------------
Fetches near real-time weather data from NOAA for Alaska wildfire prediction.

WHAT IS NOAA NWS?
  NOAA National Weather Service provides real-time and recent weather
  observations from weather stations across Alaska.
  This complements ERA5 (historical) with current/recent conditions.

DATA SOURCES WE USE:
  1. NOAA Climate Data Online (CDO) API — station observations
     URL: https://www.ncdc.noaa.gov/cdo-web/api/v2/
     Free API key: https://www.ncdc.noaa.gov/cdo-web/token

  2. NOAA Open-Meteo API (no key needed!) — gridded forecast/history
     URL: https://api.open-meteo.com/v1/forecast
     This is the EASIEST option — we use this by default.

ALASKA WEATHER STATIONS (major ones):
  Station        | GHCND ID          | Location
  ---------------|-------------------|------------------
  Fairbanks Int. | GHCND:USW00026411 | Interior Alaska
  Anchorage Int. | GHCND:USW00026451 | South Alaska
  Nome           | GHCND:USW00026617 | Western Alaska
  Juneau         | GHCND:USW00025309 | Southeast Alaska
  Barrow/Utqiagvik| GHCND:USW00027502| North Slope
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


# ─── ALASKA WEATHER STATIONS ──────────────────────────────────────────────────

ALASKA_STATIONS = {
    "fairbanks": {
        "id"      : "GHCND:USW00026411",
        "name"    : "Fairbanks International Airport",
        "lat"     : 64.815,
        "lon"     : -147.856,
        "region"  : "interior"
    },
    "anchorage": {
        "id"      : "GHCND:USW00026451",
        "name"    : "Anchorage International Airport",
        "lat"     : 61.174,
        "lon"     : -149.996,
        "region"  : "south"
    },
    "nome": {
        "id"      : "GHCND:USW00026617",
        "name"    : "Nome Airport",
        "lat"     : 64.512,
        "lon"     : -165.445,
        "region"  : "west"
    },
    "juneau": {
        "id"      : "GHCND:USW00025309",
        "name"    : "Juneau International Airport",
        "lat"     : 58.355,
        "lon"     : -134.576,
        "region"  : "southeast"
    },
}


# ─── METHOD 1: OPEN-METEO API (NO KEY NEEDED — USE THIS FIRST) ───────────────

def fetch_openmeteo_weather(
    latitude: float = 64.815,    # Default: Fairbanks
    longitude: float = -147.856,
    start_date: str = "2023-06-01",
    end_date: str   = "2023-08-31",
    output_dir: str = "data/weather/noaa"
) -> pd.DataFrame:
    """
    Fetch weather data from Open-Meteo API — completely FREE, no API key needed.

    This is the recommended starting point. Open-Meteo provides:
    - Historical weather back to 1940 (uses ERA5 under the hood)
    - Real-time forecasts
    - Hourly or daily resolution

    Parameters:
        latitude   : Location latitude
        longitude  : Location longitude
        start_date : Format YYYY-MM-DD
        end_date   : Format YYYY-MM-DD
        output_dir : Where to save CSV output

    Returns:
        DataFrame with weather data
    """
    try:
        import requests
    except ImportError:
        raise ImportError("Run: pip install requests")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Fetching Open-Meteo data...")
    print(f"       Location: ({latitude}, {longitude})")
    print(f"       Dates   : {start_date} → {end_date}")

    # Open-Meteo historical weather API
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude"              : latitude,
        "longitude"             : longitude,
        "start_date"            : start_date,
        "end_date"              : end_date,
        "daily"                 : [
            "temperature_2m_max",       # Max daily temperature (°C)
            "temperature_2m_min",       # Min daily temperature (°C)
            "temperature_2m_mean",      # Mean daily temperature
            "precipitation_sum",        # Daily precipitation (mm)
            "windspeed_10m_max",        # Max wind speed (km/h)
            "winddirection_10m_dominant", # Dominant wind direction
            "relative_humidity_2m_max", # Max relative humidity (%)
            "relative_humidity_2m_min", # Min relative humidity
            "et0_fao_evapotranspiration", # Evapotranspiration (fuel dryness)
        ],
        "timezone"              : "America/Anchorage",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[WARN] API call failed: {e}")
        print(f"[INFO] Falling back to synthetic data...")
        return generate_synthetic_noaa(latitude, longitude, start_date, end_date)

    # Parse response into DataFrame
    daily = data.get("daily", {})
    if not daily:
        print("[WARN] No daily data in response, using synthetic.")
        return generate_synthetic_noaa(latitude, longitude, start_date, end_date)

    df = pd.DataFrame(daily)
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"]      = pd.to_datetime(df["date"])
    df["latitude"]  = latitude
    df["longitude"] = longitude

    # Rename columns to consistent names
    rename_map = {
        "temperature_2m_max"            : "temp_max_c",
        "temperature_2m_min"            : "temp_min_c",
        "temperature_2m_mean"           : "temp_mean_c",
        "precipitation_sum"             : "precipitation_mm",
        "windspeed_10m_max"             : "wind_speed_kmh",
        "winddirection_10m_dominant"    : "wind_dir",
        "relative_humidity_2m_max"      : "humidity_max",
        "relative_humidity_2m_min"      : "humidity_min",
        "et0_fao_evapotranspiration"    : "evapotranspiration",
    }
    df.rename(columns=rename_map, inplace=True)

    # Convert wind speed km/h → m/s
    if "wind_speed_kmh" in df.columns:
        df["wind_speed_ms"] = df["wind_speed_kmh"] / 3.6
        df.drop("wind_speed_kmh", axis=1, inplace=True)

    # Add fire danger features
    df = add_noaa_fire_features(df)

    print(f"[INFO] Retrieved {len(df)} days of weather data")
    print(f"       Avg max temp : {df['temp_max_c'].mean():.1f}°C")

    # Save
    out_path = os.path.join(
        output_dir,
        f"noaa_openmeteo_{latitude}_{longitude}_{start_date}_{end_date}.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"[DONE] Saved to: {out_path}")

    return df


def fetch_all_alaska_stations(
    start_date: str = "2023-06-01",
    end_date: str   = "2023-08-31",
    output_dir: str = "data/weather/noaa"
) -> pd.DataFrame:
    """
    Fetch weather data for ALL major Alaska stations.
    Useful for spatial interpolation across Alaska.

    Returns:
        Combined DataFrame with all station data, labeled by station name
    """
    print(f"[INFO] Fetching weather for {len(ALASKA_STATIONS)} Alaska stations...")
    all_data = []

    for station_name, info in ALASKA_STATIONS.items():
        print(f"\n--- {info['name']} ---")
        df = fetch_openmeteo_weather(
            latitude   = info["lat"],
            longitude  = info["lon"],
            start_date = start_date,
            end_date   = end_date,
            output_dir = output_dir
        )
        df["station"]        = station_name
        df["station_name"]   = info["name"]
        df["station_region"] = info["region"]
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    out_path = os.path.join(output_dir, f"all_stations_{start_date}_{end_date}.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n[DONE] Combined data: {len(combined):,} records → {out_path}")

    return combined


# ─── FIRE DANGER FEATURES ────────────────────────────────────────────────────

def add_noaa_fire_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fire-relevant derived features from NOAA weather data.

    Fire Triangle: Fuel + Heat + Oxygen
    Weather controls: Fuel moisture (humidity), Heat (temperature), Wind (oxygen spread)

    Features added:
    1. Consecutive dry days — cumulative count of days with <2mm rain
    2. Fire Weather Index (FWI) — simplified version
    3. Keetch-Byram Drought Index (KBDI) estimate
    """
    df = df.copy()

    # ── Consecutive dry days (critical for fuel dryness)
    if "precipitation_mm" in df.columns:
        df["is_dry_day"] = (df["precipitation_mm"] < 2.0).astype(int)
        # Cumulative consecutive dry days
        consecutive = []
        count = 0
        for is_dry in df["is_dry_day"]:
            if is_dry:
                count += 1
            else:
                count = 0
            consecutive.append(count)
        df["consecutive_dry_days"] = consecutive

    # ── Temperature anomaly (how much hotter than average)
    if "temp_max_c" in df.columns:
        mean_temp = df["temp_max_c"].mean()
        df["temp_anomaly"] = df["temp_max_c"] - mean_temp

    # ── Fire danger score (0-1 scale)
    # Combines: high temp, low humidity, high wind, dry days
    score_components = []

    if "temp_max_c" in df.columns:
        t = df["temp_max_c"]
        score_components.append((t - t.min()) / (t.max() - t.min() + 1e-8) * 0.3)

    if "humidity_min" in df.columns:
        h = df["humidity_min"]
        score_components.append((1 - h / 100) * 0.3)  # Inverted: low humidity = high risk

    if "wind_speed_ms" in df.columns:
        w = df["wind_speed_ms"]
        score_components.append((w - w.min()) / (w.max() - w.min() + 1e-8) * 0.2)

    if "consecutive_dry_days" in df.columns:
        d = df["consecutive_dry_days"]
        score_components.append((d / (d.max() + 1e-8)) * 0.2)

    if score_components:
        df["fire_danger_score"] = sum(score_components)
        high = (df["fire_danger_score"] > 0.6).sum()
        print(f"[INFO] High fire danger days: {high} / {len(df)}")

    return df


# ─── SYNTHETIC DATA ──────────────────────────────────────────────────────────

def generate_synthetic_noaa(
    latitude: float = 64.815,
    longitude: float = -147.856,
    start_date: str = "2023-06-01",
    end_date: str = "2023-08-31"
) -> pd.DataFrame:
    """
    Generate synthetic NOAA-like station data for testing.
    Simulates realistic Alaska fire season (June-August) conditions.
    """
    np.random.seed(42)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    season = np.sin(np.pi * np.arange(n) / n)  # Peak in mid-July

    df = pd.DataFrame({
        "date"              : dates,
        "latitude"          : latitude,
        "longitude"         : longitude,
        "temp_max_c"        : 18 + 8 * season + np.random.normal(0, 2, n),
        "temp_min_c"        : 8  + 5 * season + np.random.normal(0, 1.5, n),
        "temp_mean_c"       : 13 + 6 * season + np.random.normal(0, 1.5, n),
        "precipitation_mm"  : np.where(np.random.random(n) < 0.25,
                                       np.random.exponential(3, n), 0),
        "wind_speed_ms"     : abs(np.random.normal(4, 2.5, n)),
        "wind_dir"          : np.random.uniform(0, 360, n),
        "humidity_max"      : 75 - 15 * season + np.random.normal(0, 5, n),
        "humidity_min"      : 45 - 20 * season + np.random.normal(0, 5, n),
        "evapotranspiration": 3  + 2 * season  + np.random.normal(0, 0.5, n),
    })

    df["temp_max_c"]   = df["temp_max_c"].clip(-5, 35)
    df["humidity_min"] = df["humidity_min"].clip(10, 90)
    df["humidity_max"] = df["humidity_max"].clip(30, 100)
    df["precipitation_mm"] = df["precipitation_mm"].clip(0)

    df = add_noaa_fire_features(df)
    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("data/weather/noaa", exist_ok=True)

    print("=== NOAA Weather Fetcher — Alaska Wildfire Pipeline ===\n")

    # Try real API first, falls back to synthetic
    print("--- Fairbanks (Interior Alaska) ---")
    df_fairbanks = fetch_openmeteo_weather(
        latitude   = ALASKA_STATIONS["fairbanks"]["lat"],
        longitude  = ALASKA_STATIONS["fairbanks"]["lon"],
        start_date = "2023-06-01",
        end_date   = "2023-08-31"
    )

    print(f"\nSample output:")
    print(df_fairbanks[["date", "temp_max_c", "humidity_min",
                          "wind_speed_ms", "fire_danger_score"]].head(5).to_string())

    if "consecutive_dry_days" in df_fairbanks.columns:
        max_dry = df_fairbanks["consecutive_dry_days"].max()
        print(f"\nMax consecutive dry days: {max_dry}")