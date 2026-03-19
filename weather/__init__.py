"""
weather/__init__.py
-------------------
Weather data integration package for Alaska Wildfire Prediction.

Modules:
  era5_fetcher     - Historical climate reanalysis (ECMWF ERA5)
  noaa_fetcher     - Near real-time station data (NOAA / Open-Meteo)
  temporal_aligner - Align weather timeseries with satellite dates
"""

from .era5_fetcher import generate_synthetic_era5, fetch_era5_data
from .noaa_fetcher import generate_synthetic_noaa, fetch_openmeteo_weather, fetch_all_alaska_stations
from .temporal_aligner import align_weather_to_satellite, create_aligned_dataset, normalize_weather_series

__all__ = [
    "generate_synthetic_era5",
    "fetch_era5_data",
    "generate_synthetic_noaa",
    "fetch_openmeteo_weather",
    "fetch_all_alaska_stations",
    "align_weather_to_satellite",
    "create_aligned_dataset",
    "normalize_weather_series",
]