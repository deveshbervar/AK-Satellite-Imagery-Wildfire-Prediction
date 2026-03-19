"""
era5_fetcher.py
---------------
Fetches ERA5 climate reanalysis data for Alaska wildfire prediction.

WHAT IS ERA5?
  ERA5 is ECMWF's (European Centre for Medium-Range Weather Forecasts)
  historical climate dataset. It covers 1940 to present at hourly resolution.
  For wildfire prediction we need:
    - Temperature (high temp = dry conditions = fire risk)
    - Wind speed (spreads fire)
    - Relative humidity (low humidity = dry fuel = fire risk)
    - Precipitation (recent rain = lower risk)

WHY ERA5 FOR ALASKA?
  Alaska has very few weather stations in remote areas.
  ERA5 provides complete spatial coverage via reanalysis (model + observations).
  The README mentions "past decades of weather data for almost 30 years" —
  that's ERA5.

HOW TO GET ACCESS:
  1. Register at: https://cds.climate.copernicus.eu/user/register
  2. After login, go to: https://cds.climate.copernicus.eu/api-how-to
  3. Create file: C:/Users/YourName/.cdsapirc with:
       url: https://cds.climate.copernicus.eu/api/v2
       key: YOUR_UID:YOUR_API_KEY

VARIABLES WE FETCH:
  Variable              | ERA5 name                    | Why
  ----------------------|------------------------------|---------------------------
  2m Temperature        | 2m_temperature               | High temp = fire risk
  10m Wind U-component  | 10m_u_component_of_wind      | Wind spreads fire
  10m Wind V-component  | 10m_v_component_of_wind      | Wind direction
  Relative Humidity     | (derived from dewpoint)      | Dry air = fire risk
  Total Precipitation   | total_precipitation          | Recent rain lowers risk
  Dewpoint Temperature  | 2m_dewpoint_temperature      | Used to calc humidity
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


# ─── ALASKA BOUNDING BOX ─────────────────────────────────────────────────────
# Same regions as satellite downloader for spatial alignment
ALASKA_REGIONS = {
    "interior": {"north": 66.5, "west": -152.0, "south": 63.0, "east": -145.0},
    "south":    {"north": 62.0, "west": -155.0, "south": 59.0, "east": -148.0},
    "full":     {"north": 71.5, "west": -168.0, "south": 54.5, "east": -130.0},
}

# ERA5 variables needed for fire risk
FIRE_WEATHER_VARIABLES = [
    "2m_temperature",           # K → convert to °C
    "2m_dewpoint_temperature",  # K → used to calc relative humidity
    "10m_u_component_of_wind",  # m/s east-west wind
    "10m_v_component_of_wind",  # m/s north-south wind
    "total_precipitation",      # metres per hour
]


# ─── ERA5 DOWNLOAD VIA CDS API ───────────────────────────────────────────────

def fetch_era5_data(
    region: str = "interior",
    start_year: int = 2020,
    end_year: int = 2023,
    months: list = [6, 7, 8],   # June, July, August = Alaska fire season
    output_dir: str = "data/weather/era5"
) -> str:
    """
    Download ERA5 reanalysis data for Alaska using the CDS API.

    Parameters:
        region     : One of 'interior', 'south', 'full'
        start_year : First year to download
        end_year   : Last year to download (inclusive)
        months     : List of months (1-12). Default = fire season (6,7,8)
        output_dir : Directory to save downloaded NetCDF files

    Returns:
        Path to downloaded NetCDF file

    Setup required:
        pip install cdsapi
        Create ~/.cdsapirc with your API credentials
    """
    try:
        import cdsapi
    except ImportError:
        raise ImportError("Run: pip install cdsapi")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bbox = ALASKA_REGIONS[region]
    area = [bbox["north"], bbox["west"], bbox["south"], bbox["east"]]

    years  = [str(y) for y in range(start_year, end_year + 1)]
    months_str = [str(m).zfill(2) for m in months]
    days   = [str(d).zfill(2) for d in range(1, 32)]
    times  = ["00:00", "06:00", "12:00", "18:00"]  # Every 6 hours

    output_file = os.path.join(
        output_dir,
        f"era5_alaska_{region}_{start_year}_{end_year}_fire_season.nc"
    )

    print(f"[INFO] Downloading ERA5 data...")
    print(f"       Region : {region} {area}")
    print(f"       Years  : {start_year} - {end_year}")
    print(f"       Months : {months} (fire season)")
    print(f"       Output : {output_file}")
    print(f"[INFO] This download may take 10-30 minutes...")

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": FIRE_WEATHER_VARIABLES,
            "year": years,
            "month": months_str,
            "day": days,
            "time": times,
            "area": area,
            "format": "netcdf",
        },
        output_file
    )

    print(f"[DONE] ERA5 data saved to: {output_file}")
    return output_file


# ─── PROCESS ERA5 DATA ───────────────────────────────────────────────────────

def process_era5_file(nc_file: str, output_csv: str = None) -> pd.DataFrame:
    """
    Process downloaded ERA5 NetCDF file into a clean DataFrame.

    Converts raw ERA5 variables into fire-relevant features:
    - Temperature: Kelvin → Celsius
    - Wind speed: U+V components → speed (m/s) and direction (degrees)
    - Relative humidity: Derived from temperature and dewpoint
    - Fire Weather Index features

    Parameters:
        nc_file    : Path to ERA5 .nc file
        output_csv : If given, saves processed data as CSV

    Returns:
        DataFrame with columns: time, lat, lon, temp_c, wind_speed,
                                wind_dir, rel_humidity, precipitation
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("Run: pip install xarray netCDF4")

    print(f"[INFO] Processing ERA5 file: {nc_file}")
    ds = xr.open_dataset(nc_file)

    print(f"       Variables : {list(ds.data_vars)}")
    print(f"       Time range: {ds.time.values[0]} → {ds.time.values[-1]}")
    print(f"       Shape     : {dict(ds.dims)}")

    # Convert to DataFrame
    df = ds.to_dataframe().reset_index()

    # ── Temperature: Kelvin → Celsius
    if "t2m" in df.columns:
        df["temp_c"] = df["t2m"] - 273.15
        df.drop("t2m", axis=1, inplace=True)

    # ── Dewpoint: Kelvin → Celsius
    if "d2m" in df.columns:
        df["dewpoint_c"] = df["d2m"] - 273.15
        df.drop("d2m", axis=1, inplace=True)

    # ── Relative Humidity from temperature and dewpoint
    # Formula: Magnus approximation
    # RH = 100 * exp(17.625 * Td / (243.04 + Td)) / exp(17.625 * T / (243.04 + T))
    if "temp_c" in df.columns and "dewpoint_c" in df.columns:
        T  = df["temp_c"]
        Td = df["dewpoint_c"]
        df["rel_humidity"] = 100 * (
            np.exp(17.625 * Td / (243.04 + Td)) /
            np.exp(17.625 * T  / (243.04 + T))
        )
        df["rel_humidity"] = df["rel_humidity"].clip(0, 100)

    # ── Wind Speed and Direction from U/V components
    if "u10" in df.columns and "v10" in df.columns:
        u = df["u10"]
        v = df["v10"]
        df["wind_speed"] = np.sqrt(u**2 + v**2)                    # m/s
        df["wind_dir"]   = (np.degrees(np.arctan2(u, v)) + 360) % 360  # degrees
        df.drop(["u10", "v10"], axis=1, inplace=True)

    # ── Precipitation: metres → mm
    if "tp" in df.columns:
        df["precipitation_mm"] = df["tp"] * 1000
        df.drop("tp", axis=1, inplace=True)

    # ── Fire danger features
    df = add_fire_danger_features(df)

    # Clean up
    df = df.dropna(subset=["temp_c"]) if "temp_c" in df.columns else df
    df = df.reset_index(drop=True)

    print(f"[INFO] Processed {len(df):,} records")

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[DONE] Saved to: {output_csv}")

    return df


def add_fire_danger_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived fire danger features used in wildfire risk models.

    Features added:
    1. Heat Index  : Combines temp + humidity → perceived heat stress on fuel
    2. Vapor Pressure Deficit (VPD): Key indicator of fuel dryness
       VPD > 2 kPa → HIGH fire danger
    3. Hot-Dry-Windy (HDW) Index: Simple composite risk score
    """
    if "temp_c" in df.columns and "rel_humidity" in df.columns:
        T  = df["temp_c"]
        RH = df["rel_humidity"]

        # ── Vapor Pressure Deficit (VPD) in kPa
        # Saturated vapor pressure (Tetens formula)
        e_sat = 0.6108 * np.exp(17.27 * T / (T + 237.3))
        e_act = e_sat * (RH / 100)
        df["vpd_kpa"] = (e_sat - e_act).clip(0)

        # ── Hot-Dry-Windy Index (simple composite)
        # Higher score = more dangerous fire weather
        if "wind_speed" in df.columns:
            temp_norm  = (T - T.min()) / (T.max() - T.min() + 1e-8)
            dry_norm   = 1 - (RH / 100)
            wind_norm  = (df["wind_speed"] - df["wind_speed"].min()) / \
                         (df["wind_speed"].max() - df["wind_speed"].min() + 1e-8)
            df["hdw_index"] = (temp_norm * 0.4 + dry_norm * 0.4 + wind_norm * 0.2)

    return df


# ─── SYNTHETIC DATA FOR TESTING ──────────────────────────────────────────────

def generate_synthetic_era5(
    region: str = "interior",
    n_days: int = 90,   # 3 months = fire season
    start_date: str = "2023-06-01"
) -> pd.DataFrame:
    """
    Generate synthetic ERA5-like weather data for testing WITHOUT downloading.

    Simulates realistic Alaska summer weather patterns:
    - Temperature: 10-25°C with daily cycles
    - Humidity: 40-80% (lower = more fire risk)
    - Wind: 0-15 m/s
    - Precipitation: Mostly dry with occasional rain

    Parameters:
        region     : One of 'interior', 'south', 'full'
        n_days     : Number of days to simulate
        start_date : Start date string (YYYY-MM-DD)

    Returns:
        DataFrame with realistic weather features
    """
    np.random.seed(42)
    bbox = ALASKA_REGIONS[region]

    # Create spatial grid (5 lat/lon points for simplicity)
    lats = np.linspace(bbox["south"], bbox["north"], 5)
    lons = np.linspace(bbox["west"],  bbox["east"],  5)

    records = []
    base_date = pd.Timestamp(start_date)

    for day in range(n_days):
        current_date = base_date + pd.Timedelta(days=day)

        # Seasonal temperature curve (peaks in late July)
        season_factor = np.sin(np.pi * day / n_days)

        for lat in lats:
            for lon in lons:
                # Temperature: 10-28°C in summer Alaska interior
                temp_c = 15 + 10 * season_factor + np.random.normal(0, 2)

                # Humidity: lower when hotter (inverse relationship)
                rel_humidity = 65 - 20 * season_factor + np.random.normal(0, 8)
                rel_humidity = np.clip(rel_humidity, 20, 95)

                # Wind: random with slight seasonal pattern
                wind_speed = abs(np.random.normal(4, 3))
                wind_dir   = np.random.uniform(0, 360)

                # Precipitation: mostly dry in interior Alaska fire season
                precip = max(0, np.random.exponential(0.5) if np.random.random() < 0.2 else 0)

                record = {
                    "time"          : current_date,
                    "latitude"      : round(lat, 2),
                    "longitude"     : round(lon, 2),
                    "temp_c"        : round(temp_c, 2),
                    "rel_humidity"  : round(rel_humidity, 2),
                    "wind_speed"    : round(wind_speed, 2),
                    "wind_dir"      : round(wind_dir, 1),
                    "precipitation_mm": round(precip, 3),
                    "dewpoint_c"    : round(temp_c - (100 - rel_humidity) / 5, 2),
                }
                records.append(record)

    df = pd.DataFrame(records)
    df = add_fire_danger_features(df)

    print(f"[INFO] Generated synthetic ERA5 data")
    print(f"       Records   : {len(df):,}")
    print(f"       Date range: {df['time'].min()} → {df['time'].max()}")
    print(f"       Avg temp  : {df['temp_c'].mean():.1f}°C")
    print(f"       Avg RH    : {df['rel_humidity'].mean():.1f}%")
    if "vpd_kpa" in df.columns:
        print(f"       Avg VPD   : {df['vpd_kpa'].mean():.3f} kPa")

    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ERA5 Weather Fetcher — Alaska Wildfire Pipeline ===")
    print("Running synthetic test (no credentials needed)...\n")

    df = generate_synthetic_era5(region="interior", n_days=90)

    print(f"\nSample data:")
    print(df.head(3).to_string())

    print(f"\nFire danger summary:")
    if "vpd_kpa" in df.columns:
        high_vpd = (df["vpd_kpa"] > 1.0).sum()
        print(f"  High VPD days (>1.0 kPa): {high_vpd:,} / {len(df):,} records")
    if "hdw_index" in df.columns:
        high_hdw = (df["hdw_index"] > 0.6).sum()
        print(f"  High HDW days (>0.6)    : {high_hdw:,} / {len(df):,} records")

    df.to_csv("data/weather/era5_synthetic_test.csv", index=False)
    print(f"\n[DONE] Saved to data/weather/era5_synthetic_test.csv")