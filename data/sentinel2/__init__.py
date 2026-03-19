"""
data/sentinel2/__init__.py
--------------------------
Makes this directory a Python package so you can import like:
  from data.sentinel2.cloud_mask import mask_clouds_in_file
  from data.sentinel2.band_extractor import compute_ndvi
"""

from .cloud_mask import create_cloud_mask, apply_cloud_mask, mask_clouds_in_file
from .band_extractor import extract_bands, compute_all_indices, compute_fire_risk_score
from .normalizer import normalize_percentile, extract_patches, prepare_for_model

__all__ = [
    "create_cloud_mask",
    "apply_cloud_mask",
    "mask_clouds_in_file",
    "extract_bands",
    "compute_all_indices",
    "compute_fire_risk_score",
    "normalize_percentile",
    "extract_patches",
    "prepare_for_model",
]