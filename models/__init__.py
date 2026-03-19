"""
models/__init__.py
------------------
Model package for Alaska Wildfire Prediction.

Models:
  SentinelCNN         - CNN for Sentinel-2 spatial feature extraction
  WeatherLSTM         - LSTM for weather time series
  WeatherLSTMWithAttention - LSTM with attention mechanism
  WildfireRiskModel   - Hybrid CNN-LSTM for fire risk classification
"""

from .cnn_feature_extractor import SentinelCNN, ConvBlock
from .lstm_weather import WeatherLSTM, WeatherLSTMWithAttention
from .hybrid_model import WildfireRiskModel, create_model, compute_loss, compute_metrics

__all__ = [
    "SentinelCNN",
    "ConvBlock",
    "WeatherLSTM",
    "WeatherLSTMWithAttention",
    "WildfireRiskModel",
    "create_model",
    "compute_loss",
    "compute_metrics",
]