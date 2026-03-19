"""
downloader.py
-------------
Downloads Sentinel-2 satellite imagery for Alaska wildfire prediction.
 
Two methods available:
  1. sentinelsat  — uses Copernicus Open Access Hub (free account needed)
  2. Google Earth Engine (GEE) — uses ee Python API (free, needs signup)
 
For beginners: Use Method 2 (GEE) — easier setup, no download quota limits.
 
HOW TO GET CREDENTIALS:
  - Copernicus Hub: https://scihub.copernicus.eu/dhus/#/self-registration
  - Google Earth Engine: https://earthengine.google.com/signup/
"""
 
import os
import json
from datetime import datetime
from pathlib import Path
 
# ─── ALASKA BOUNDING BOX ────────────────────────────────────────────────────
# These coordinates cover the interior Alaska region where wildfires are common
# Format: (min_longitude, min_latitude, max_longitude, max_latitude)
ALASKA_BBOX = {
    "interior": (-152.0, 63.0, -145.0, 66.5),   # Fairbanks / interior region
    "south":    (-155.0, 59.0, -148.0, 62.0),   # Kenai Peninsula
    "full":     (-168.0, 54.5, -130.0, 71.5),   # All of Alaska (very large!)
}
 
# ─── METHOD 1: SENTINELSAT ───────────────────────────────────────────────────
 
def download_via_sentinelsat(
    username: str,
    password: str,
    region: str = "interior",
    start_date: str = "20230601",   # Fire season starts June
    end_date: str   = "20230831",   # Fire season ends August
    cloud_cover_max: int = 30,       # Max 30% cloud cover
    output_dir: str = "data/raw/sentinel2"
) -> list:
    """
    Download Sentinel-2 L2A tiles using sentinelsat library.
 
    Parameters:
        username        : Copernicus Hub username
        password        : Copernicus Hub password
        region          : One of 'interior', 'south', 'full'
        start_date      : Format YYYYMMDD
        end_date        : Format YYYYMMDD
        cloud_cover_max : Filter out tiles with more than X% clouds
        output_dir      : Where to save downloaded .zip files
 
    Returns:
        List of downloaded file paths
    """
    try:
        from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
        from shapely.geometry import box
    except ImportError:
        raise ImportError("Run: pip install sentinelsat shapely")
 
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
 
    # Connect to Copernicus Open Access Hub
    print(f"[INFO] Connecting to Copernicus Hub as '{username}'...")
    api = SentinelAPI(username, password, "https://scihub.copernicus.eu/dhus")
 
    # Define search area as Well-Known Text (WKT) polygon
    bbox = ALASKA_BBOX[region]
    # box(min_lon, min_lat, max_lon, max_lat)
    footprint = box(bbox[0], bbox[1], bbox[2], bbox[3]).wkt
 
    print(f"[INFO] Searching for Sentinel-2 tiles...")
    print(f"       Region   : {region} {bbox}")
    print(f"       Dates    : {start_date} → {end_date}")
    print(f"       Max cloud: {cloud_cover_max}%")
 
    # Query the API
    products = api.query(
        footprint,
        date=(start_date, end_date),
        platformname="Sentinel-2",
        producttype="S2MSI2A",          # L2A = atmospherically corrected
        cloudcoverpercentage=(0, cloud_cover_max),
    )
 
    print(f"[INFO] Found {len(products)} matching tiles.")
 
    if len(products) == 0:
        print("[WARN] No products found. Try expanding date range or cloud cover %.")
        return []
 
    # Download all found products
    downloaded = []
    for pid, pinfo in products.items():
        print(f"[INFO] Downloading: {pinfo['title']} ({pinfo['size']})")
        api.download(pid, directory_path=output_dir)
        downloaded.append(os.path.join(output_dir, pinfo["title"] + ".zip"))
 
    print(f"\n[DONE] Downloaded {len(downloaded)} tiles to '{output_dir}'")
    return downloaded
 
 
# ─── METHOD 2: GOOGLE EARTH ENGINE (RECOMMENDED FOR BEGINNERS) ──────────────
 
def download_via_gee(
    region: str = "interior",
    start_date: str = "2023-06-01",
    end_date: str   = "2023-08-31",
    cloud_cover_max: int = 30,
    output_dir: str = "data/raw/sentinel2"
) -> str:
    """
    Export Sentinel-2 imagery from Google Earth Engine to local GeoTIFF.
 
    SETUP (one-time):
        1. Sign up at https://earthengine.google.com
        2. Run in terminal: earthengine authenticate
        3. This opens a browser — log in with your Google account
 
    Parameters:
        region          : One of 'interior', 'south', 'full'
        start_date      : Format YYYY-MM-DD
        end_date        : Format YYYY-MM-DD
        cloud_cover_max : Max cloud cover percentage to include
        output_dir      : Where to save the exported GeoTIFF
 
    Returns:
        Path to the exported .tif file
    """
    try:
        import ee
        import geemap   # pip install geemap
    except ImportError:
        raise ImportError("Run: pip install earthengine-api geemap")
 
    Path(output_dir).mkdir(parents=True, exist_ok=True)
 
    # Authenticate (only needed first time — opens browser)
    try:
        ee.Initialize()
        print("[INFO] GEE initialized successfully.")
    except Exception:
        print("[INFO] Running GEE authentication...")
        ee.Authenticate()
        ee.Initialize()
 
    # Define Alaska region of interest
    bbox = ALASKA_BBOX[region]
    roi = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])
 
    print(f"[INFO] Loading Sentinel-2 collection...")
    print(f"       Region: {region} | Dates: {start_date} → {end_date}")
 
    # Load Sentinel-2 Surface Reflectance collection
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_max))
        .select(["B2", "B3", "B4", "B8", "B11", "B12", "QA60"])
        # B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR1, B12=SWIR2, QA60=cloud mask
    )
 
    count = collection.size().getInfo()
    print(f"[INFO] Found {count} images after filtering.")
 
    if count == 0:
        print("[WARN] No images found. Try different dates or region.")
        return None
 
    # Create median composite (reduces cloud artifacts)
    composite = collection.median().clip(roi)
 
    # Export as GeoTIFF
    output_path = os.path.join(output_dir, f"alaska_{region}_{start_date}_{end_date}.tif")
    print(f"[INFO] Exporting composite to: {output_path}")
    print("[INFO] This may take a few minutes...")
 
    geemap.ee_export_image(
        composite,
        filename=output_path,
        scale=10,           # 10m resolution (Sentinel-2 native)
        region=roi,
        file_per_band=False
    )
 
    print(f"[DONE] Saved to: {output_path}")
    return output_path
 
 
# ─── MAIN: Quick test ────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("=== Sentinel-2 Downloader ===")
    print("Choose method:")
    print("  1. Copernicus Hub (sentinelsat)")
    print("  2. Google Earth Engine (recommended)")
    choice = input("Enter 1 or 2: ").strip()
 
    if choice == "1":
        user = input("Copernicus username: ")
        pwd  = input("Copernicus password: ")
        download_via_sentinelsat(user, pwd)
    elif choice == "2":
        download_via_gee()
    else:
        print("Invalid choice.")
 