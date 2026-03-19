"""
hybrid_model.py
---------------
Hybrid CNN-LSTM model combining satellite imagery and weather time series
for Alaska wildfire risk prediction.

THIS IS THE CORE MODEL the project README describes:
  "A hybrid model such as CNN-LSTM that analyzes satellite data
   and time-series weather trends"

HOW IT WORKS:
  1. CNN processes satellite patch → spatial features (128-dim)
  2. LSTM processes weather series → temporal features (64-dim)
  3. Features are CONCATENATED → (192-dim combined vector)
  4. Fully connected layers → fire risk classification

PREDICTION CLASSES:
  0 = Low Risk    ("No Risk" in README)
  1 = Moderate Risk
  2 = High Risk   ("High Fire Risk" in README)

FULL ARCHITECTURE:

  Sentinel-2 patch          Weather time series
  (B, 6, 64, 64)            (B, 14, 8)
       ↓                          ↓
  SentinelCNN               WeatherLSTM
       ↓                          ↓
  spatial_features(128)     weather_features(64)
       ↓                          ↓
       └──────────┬───────────────┘
                  ↓
          concat → (192,)
                  ↓
          Linear(192 → 96) + ReLU + Dropout
                  ↓
          Linear(96 → 48) + ReLU + Dropout
                  ↓
          Linear(48 → 3)  ← 3 risk classes
                  ↓
          Softmax → probabilities
          [P(low), P(moderate), P(high)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cnn_feature_extractor import SentinelCNN
from lstm_weather import WeatherLSTM, WeatherLSTMWithAttention


class WildfireRiskModel(nn.Module):
    """
    Hybrid CNN-LSTM model for Alaska wildfire risk prediction.

    Parameters:
        n_satellite_bands : Input bands from Sentinel-2 (default 6)
        n_weather_features: Weather features per day (default 8)
        n_lookback_days   : Days of weather history (default 14)
        cnn_feature_dim   : CNN output size (default 128)
        lstm_hidden_size  : LSTM hidden state size (default 64)
        lstm_feature_dim  : LSTM output size (default 64)
        n_classes         : Number of risk classes (default 3)
        dropout_rate      : Regularization dropout (default 0.3)
        use_attention     : Use attention LSTM for better interpretability

    Usage:
        model = WildfireRiskModel()

        # Inputs
        sat_patch = torch.randn(32, 6, 64, 64)    # satellite patches
        weather   = torch.randn(32, 14, 8)         # weather series

        # Predict
        output = model(sat_patch, weather)
        # output['probabilities'] shape: (32, 3) = [P(low), P(mod), P(high)]
        # output['predicted_class'] shape: (32,)  = 0, 1, or 2
    """

    def __init__(
        self,
        n_satellite_bands  : int = 6,
        n_weather_features : int = 8,
        n_lookback_days    : int = 14,
        cnn_feature_dim    : int = 128,
        lstm_hidden_size   : int = 64,
        lstm_feature_dim   : int = 64,
        n_classes          : int = 3,
        dropout_rate       : float = 0.3,
        use_attention      : bool = False
    ):
        super(WildfireRiskModel, self).__init__()

        self.n_classes       = n_classes
        self.cnn_feature_dim = cnn_feature_dim
        self.lstm_feature_dim= lstm_feature_dim

        # ── Branch 1: CNN for satellite spatial features
        self.cnn = SentinelCNN(
            in_channels  = n_satellite_bands,
            feature_dim  = cnn_feature_dim,
            dropout_rate = dropout_rate
        )

        # ── Branch 2: LSTM for weather temporal features
        LSTMClass = WeatherLSTMWithAttention if use_attention else WeatherLSTM
        self.lstm = LSTMClass(
            input_size   = n_weather_features,
            hidden_size  = lstm_hidden_size,
            feature_dim  = lstm_feature_dim,
        )

        # ── Combined feature size after concatenation
        combined_dim = cnn_feature_dim + lstm_feature_dim   # 128 + 64 = 192

        # ── Fusion and classification layers
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(96, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(48, n_classes)
            # Note: No softmax here — CrossEntropyLoss handles it during training
            # For inference, we apply softmax manually
        )

        # ── Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Xavier initialization for linear layers.
        Helps training converge faster by setting good initial weights.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, satellite_patch: torch.Tensor, weather_series: torch.Tensor) -> dict:
        """
        Forward pass through the hybrid model.

        Parameters:
            satellite_patch : (batch, n_bands, H, W) — Sentinel-2 patches
            weather_series  : (batch, n_days, n_features) — weather time series

        Returns:
            Dictionary with:
              'logits'          : Raw scores (batch, n_classes) — for loss calculation
              'probabilities'   : Softmax probabilities (batch, n_classes)
              'predicted_class' : Argmax class (batch,) — 0=low, 1=mod, 2=high
              'spatial_features': CNN output (batch, cnn_feature_dim)
              'temporal_features': LSTM output (batch, lstm_feature_dim)
        """
        # ── Branch 1: CNN extracts spatial features from satellite patch
        spatial_features = self.cnn(satellite_patch)    # (B, 128)

        # ── Branch 2: LSTM extracts temporal features from weather
        temporal_features = self.lstm(weather_series)   # (B, 64)

        # ── Fuse: concatenate both feature vectors
        combined = torch.cat([spatial_features, temporal_features], dim=1)  # (B, 192)

        # ── Classify into fire risk levels
        logits = self.classifier(combined)              # (B, 3)

        # ── Compute probabilities (for inference)
        probabilities = F.softmax(logits, dim=1)        # (B, 3)

        # ── Get predicted class
        predicted_class = torch.argmax(probabilities, dim=1)  # (B,)

        return {
            "logits"           : logits,
            "probabilities"    : probabilities,
            "predicted_class"  : predicted_class,
            "spatial_features" : spatial_features,
            "temporal_features": temporal_features,
        }

    def predict_risk_label(self, satellite_patch: torch.Tensor,
                           weather_series: torch.Tensor) -> list:
        """
        Convenience method — returns human-readable risk labels.

        Returns:
            List of strings: "Low Risk", "Moderate Risk", or "High Risk"
        """
        labels = ["Low Risk", "Moderate Risk", "High Risk"]
        self.eval()
        with torch.no_grad():
            output = self.forward(satellite_patch, weather_series)
        return [labels[c.item()] for c in output["predicted_class"]]

    def get_model_info(self) -> dict:
        """Return model summary information."""
        total_params    = sum(p.numel() for p in self.parameters())
        trainable       = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn_params      = sum(p.numel() for p in self.cnn.parameters())
        lstm_params     = sum(p.numel() for p in self.lstm.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())

        return {
            "total_parameters"     : total_params,
            "trainable_parameters" : trainable,
            "cnn_parameters"       : cnn_params,
            "lstm_parameters"      : lstm_params,
            "classifier_parameters": classifier_params,
            "input_satellite"      : f"(batch, {self.cnn.in_channels}, 64, 64)",
            "input_weather"        : f"(batch, 14, {self.lstm.input_size})",
            "output_classes"       : self.n_classes,
            "output_labels"        : ["Low Risk", "Moderate Risk", "High Risk"],
        }


# ─── TRAINING UTILITIES ───────────────────────────────────────────────────────

def create_model(config: dict = None) -> WildfireRiskModel:
    """
    Factory function to create model with default or custom config.

    Parameters:
        config : Optional dict to override defaults

    Returns:
        Initialized WildfireRiskModel
    """
    defaults = {
        "n_satellite_bands"  : 6,
        "n_weather_features" : 8,
        "n_lookback_days"    : 14,
        "cnn_feature_dim"    : 128,
        "lstm_hidden_size"   : 64,
        "lstm_feature_dim"   : 64,
        "n_classes"          : 3,
        "dropout_rate"       : 0.3,
        "use_attention"      : False,
    }
    if config:
        defaults.update(config)
    return WildfireRiskModel(**defaults)


def compute_loss(outputs: dict, labels: torch.Tensor,
                 class_weights: torch.Tensor = None) -> torch.Tensor:
    """
    Compute cross-entropy loss for fire risk classification.

    WHY CLASS WEIGHTS?
      In Alaska fire data, "Low Risk" days vastly outnumber "High Risk" days.
      Without weights, model learns to always predict "Low Risk."
      Class weights penalize wrong predictions on rare High Risk days more.

    Parameters:
        outputs       : Model output dict from forward()
        labels        : Ground truth class indices (batch,) — 0, 1, or 2
        class_weights : Optional tensor (3,) to handle class imbalance

    Returns:
        Scalar loss tensor
    """
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion(outputs["logits"], labels)


def compute_metrics(outputs: dict, labels: torch.Tensor) -> dict:
    """
    Compute evaluation metrics for fire risk prediction.

    Returns:
        Dictionary with accuracy, per-class accuracy
    """
    predicted = outputs["predicted_class"]
    correct   = (predicted == labels).float()
    accuracy  = correct.mean().item()

    per_class = {}
    for cls_id, cls_name in enumerate(["Low", "Moderate", "High"]):
        mask = (labels == cls_id)
        if mask.sum() > 0:
            per_class[cls_name] = correct[mask].mean().item()
        else:
            per_class[cls_name] = None

    return {"accuracy": accuracy, "per_class_accuracy": per_class}


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Hybrid CNN-LSTM Wildfire Risk Model ===\n")
    print("Alaska Wildfire Prediction — Model Architecture Test\n")

    # ── Create model
    model = create_model()

    # ── Print model info
    info = model.get_model_info()
    print("Model Configuration:")
    for key, value in info.items():
        print(f"  {key:30s}: {value}")

    # ── Test forward pass
    print("\n--- Forward Pass Test ---")
    batch_size = 8

    sat_patches    = torch.randn(batch_size, 6, 64, 64)   # satellite patches
    weather_series = torch.randn(batch_size, 14, 8)        # weather time series
    dummy_labels   = torch.randint(0, 3, (batch_size,))    # fake labels for loss test

    model.eval()
    with torch.no_grad():
        outputs = model(sat_patches, weather_series)

    print(f"Satellite input shape  : {sat_patches.shape}")
    print(f"Weather input shape    : {weather_series.shape}")
    print(f"Logits shape           : {outputs['logits'].shape}")
    print(f"Probabilities shape    : {outputs['probabilities'].shape}")
    print(f"Predicted classes      : {outputs['predicted_class'].tolist()}")
    print(f"Spatial features shape : {outputs['spatial_features'].shape}")
    print(f"Temporal features shape: {outputs['temporal_features'].shape}")

    # ── Test loss computation
    print("\n--- Loss Computation Test ---")
    model.train()
    outputs_train = model(sat_patches, weather_series)
    loss = compute_loss(outputs_train, dummy_labels)
    print(f"Training loss: {loss.item():.4f}")

    # ── Test metrics
    print("\n--- Metrics Test ---")
    model.eval()
    with torch.no_grad():
        outputs_eval = model(sat_patches, weather_series)
    metrics = compute_metrics(outputs_eval, dummy_labels)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    for cls, acc in metrics["per_class_accuracy"].items():
        if acc is not None:
            print(f"  {cls} Risk accuracy: {acc:.4f}")

    # ── Test prediction labels
    print("\n--- Risk Label Prediction ---")
    labels = model.predict_risk_label(sat_patches[:3], weather_series[:3])
    for i, label in enumerate(labels):
        prob = outputs["probabilities"][i]
        print(f"  Sample {i+1}: {label} "
              f"[Low={prob[0]:.2f}, Mod={prob[1]:.2f}, High={prob[2]:.2f}]")

    print(f"\n[PASS] Hybrid CNN-LSTM model working correctly!")
    print(f"\nModel is ready for training with real Sentinel-2 + weather data.")
    print(f"Next step: collect labeled fire/no-fire samples and train.")