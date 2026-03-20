# Alaska Wildfire Prediction Using Satellite Imagery

**Mentors:** Yali Wang (ywang35 -at- alaska.edu) and Arghya Kusum Das (akdas -at- alaska.edu)

**Overview:** Given Alaska’s unique wildfire patterns, where large-scale fires occur annually in boreal forests, tundra, and remote wilderness, predicting fire-prone areas can help mitigate disasters and optimize resource allocation. The presence of vegetation (fuel) is necessary for a fire, but the determining factors are weather conditions (humidity, wind speed, temperature) and an ignition source (lightning, human activity, etc.). 
This project aims to develop a hybrid deep learning model to predict wildfire risk in Alaska by integrating optical, thermal, and synthetic aperture radar (SAR) satellite imagery with ground-based weather data.
Traditional wildfire prediction relies on weather data, historical fire records, and human observations, which can be delayed or inaccurate in remote areas like Alaska. In contrast, satellite imagery provides real-time, high-resolution insights into vegetation health, thermal anomalies, burn severity mapping, soil moisture, fuel dryness, and even cloud-penetrating fire detection.

Satellite choices:

| Satellite | Resolution | Revisit Frequency | Why Use It? |
| ------ | ------ | ------ | ------ |
| Landsat 8 & 9 (NASA/USGS) | 30m (multispectral), 100m (thermal) | 16 days | Tracks pre/post-fire vegetation and burn severity with great detail.|
| Sentinel-2 (ESA) | 10m (RGB, NIR), 20m (SWIR) | 5 days | High-resolution images for fire risk classification and early warnings. |
| MODIS (Terra/Aqua, NASA) | 250m (fire detection), 1km (thermal) | Daily | Provides historical fire perimeters and active fire locations. |
| VIIRS (Suomi NPP & NOAA-20) | 375m (fire detection), 750m (thermal) | Daily | Real-time fire monitoring, capturing active hotspots. |
| Sentinel-1 (ESA) | 5m - 20m | 6-12 days | SAR imaging for vegetation moisture & burned area mapping. |
| ALOS-2 (JAXA) | 10m - 100m | 14 days | L-band SAR for detecting dry fuel and terrain changes. |

Additional ground data sources:

1). ERA5 Climate Reanalysis (ECMWF): Provides historical & real-time temperature, wind, and humidity data.

2). NOAA NWS Weather Data: Near real-time humidity, wind, and temperature.

3). Alaska Fire Service (AFS) Wildfire Data: Historical ignition source data (lightning, human activity).

**Current Status:** This project is currently in the research stage.

**Expected Outcomes:** 
This project aims to develop a deep-learning model that predicts wildfire risk in Alaska using a combination of satellite and ground-based weather data. The expected outcome of this project would involve both the dataset preprocessing pipeline and the performance of the developed model. Especially, the dataset preprocessing would include how to process the pre-fire and post-fire images efficiently and integrate the ground-based data with satellite imagery. Expected outcomes include:

Minimum viable product (MVP): 

Fire risk classification: Given pre-fire satellite images, the model predicts the probability of a fire occurring within a defined time frame like 1 month, 3 months, or 6 months. The classifications should be "High Fire Risk," "Moderate Risk," or "No Risk."

1). Data pipeline development:

Preprocessing satellite images: Band selection, geospatial cropping, cloud removal (For this step, we are mostly interested in analyzing [Sentinel-2 data](https://dataspace.copernicus.eu/explore-data/data-collections/sentinel-data/sentinel-2));

Synthetic Aperture Radar (SAR) analysis: Extracting fuel moisture & terrain features (For this step, we are mostly interested in extracting information like vegetation density and soil moisture from [Sentinel-1 SAR data](https://dataspace.copernicus.eu/explore-data/data-collections/sentinel-data/sentinel-1));

Time-series weather data integration: Incorporating temperature, wind, and humidity. We have access to past decades of weather data for almost the past 30 years for multiple different places in Alaska.

2). Model training and prediction:

A hybrid model such as CNN-LSTM that analyzes satellite data and time-series weather trends (CNN-LSTM is just an example. We are open to multiple different types of analysis methodology);

A web-based GIS dashboard to visualize fire-prone regions in Alaska;

A report on model performance and fire risk metrics.

**Required Skills:** Python. Experience with deep learning and machine learning.

**Code Challenge:** Experience with multi-band satellite imagery, geospatial data processing (like ArcGIS Pro), and remote sensing.

**Source Code:** [https://github.com/YaliWang2019/AK-Satellite-Imagery-Wildfire-Prediction](https://github.com/YaliWang2019/AK-Satellite-Imagery-Wildfire-Prediction) (New Project)

**Discussion Forum:** [https://github.com/YaliWang2019/AK-Satellite-Imagery-Wildfire-Prediction/discussions](https://github.com/YaliWang2019/AK-Satellite-Imagery-Wildfire-Prediction/discussions)

**Effort:** 350 Hours

**Difficulty Level:** Medium/Hard

---

## Implementation Progress (Contributor: deveshbervar)

### Pipeline Architecture
```
Sentinel-2 Imagery          Weather Data (ERA5/NOAA)
(6 bands, 64x64 patches)    (14-day time series)
        ↓                           ↓
   SentinelCNN                WeatherLSTM
   (spatial features)         (temporal features)
        ↓                           ↓
        └──────── Fusion ───────────┘
                      ↓
            WildfireRiskModel
                      ↓
         Low / Moderate / High Risk
```

### Modules Implemented

| Module | Description | Status |
|--------|-------------|--------|
| `data/sentinel2/cloud_mask.py` | QA60 bitmask cloud removal | Done |
| `data/sentinel2/band_extractor.py` | NDVI, NBR, NDMI indices | Done |
| `data/sentinel2/normalizer.py` | Percentile normalization + patching | Done |
| `weather/era5_fetcher.py` | ERA5 climate reanalysis | Done |
| `weather/noaa_fetcher.py` | NOAA/Open-Meteo station data | Done |
| `weather/temporal_aligner.py` | 14-day weather-satellite alignment | Done |
| `models/cnn_feature_extractor.py` | CNN for satellite spatial features | Done |
| `models/lstm_weather.py` | LSTM for weather time series | Done |
| `models/hybrid_model.py` | Hybrid CNN-LSTM (321k parameters) | Done |

### Quick Test
```bash
pip install -r requirements.txt
python data/sentinel2/band_extractor.py   # Test satellite pipeline
python weather/noaa_fetcher.py            # Test weather pipeline
cd models && python hybrid_model.py       # Test CNN-LSTM model
```