# Sentinel-2 Preprocessing Pipeline

This module implements the core satellite imagery preprocessing pipeline for the
Alaska Wildfire Prediction project, as described in the project MVP requirements.

## What This Pipeline Does

```
Raw Sentinel-2 Tiles
        ↓
  [1. Download]      ← downloader.py
  GEE or Copernicus Hub
        ↓
  [2. Cloud Mask]    ← cloud_mask.py
  Remove cloudy pixels using QA60 band
        ↓
  [3. Band Extract]  ← band_extractor.py
  B2, B3, B4, B8, B11, B12
  NDVI, NBR, NDMI indices
  Fire risk score
        ↓
  [4. Normalize]     ← normalizer.py
  Scale to [0,1], split into patches
        ↓
  Model-Ready Input
  Shape: (N_patches, 6_bands, 64, 64)
```

## Setup

```bash
# Clone the repo
git clone https://github.com/YaliWang2019/AK-Satellite-Imagery-Wildfire-Prediction
cd AK-Satellite-Imagery-Wildfire-Prediction

# Install dependencies
pip install -r requirements.txt

# (One-time) Authenticate Google Earth Engine
earthengine authenticate
```

## Quick Start

### Run the demo notebook (no credentials needed)
```bash
jupyter notebook notebooks/sentinel2_demo.ipynb
```

### Download real data and process
```python
from data.sentinel2.downloader import download_via_gee
from data.sentinel2.cloud_mask import mask_clouds_in_file
from data.sentinel2.band_extractor import extract_bands, compute_all_indices
from data.sentinel2.normalizer import prepare_for_model

# Step 1: Download
tif_path = download_via_gee(region='interior', start_date='2023-06-01', end_date='2023-08-31')

# Step 2: Cloud mask
masked_bands, clear_mask, meta = mask_clouds_in_file(tif_path, output_tif='masked.tif')

# Step 3: Extract bands and indices
bands   = extract_bands(masked_bands)
indices = compute_all_indices(bands)

# Step 4: Normalize and patch
patches, positions, stats = prepare_for_model(masked_bands, method='percentile')
print(f'Model input shape: {patches.shape}')
```

## Vegetation Indices Computed

| Index | Formula | Wildfire Relevance |
|-------|---------|-------------------|
| NDVI  | (NIR - Red) / (NIR + Red) | Vegetation density = fuel load |
| NBR   | (NIR - SWIR2) / (NIR + SWIR2) | Burn severity detection |
| NDMI  | (NIR - SWIR1) / (NIR + SWIR1) | Fuel moisture (dry = high risk) |

## Alaska Study Region

| Region | Bounding Box | Notes |
|--------|-------------|-------|
| interior | (-152, 63, -145, 66.5) | Fairbanks area — most active fire zone |
| south | (-155, 59, -148, 62) | Kenai Peninsula |
| full | (-168, 54.5, -130, 71.5) | All of Alaska (very large) |

## Next Steps

This pipeline feeds into:
1. **Sentinel-1 SAR module** — soil moisture and vegetation structure
2. **Weather integration** — ERA5/NOAA temperature, wind, humidity
3. **CNN-LSTM model** — spatial + temporal wildfire risk prediction
4. **GIS dashboard** — interactive fire risk map of Alaska