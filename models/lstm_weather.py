"""
lstm_weather.py
---------------
LSTM model that processes weather time series for wildfire prediction.

WHY LSTM FOR WEATHER DATA?
  Weather data is a TIME SERIES — what happened yesterday affects today.
  A forest that had 14 dry days is much more dangerous than one dry day.
  LSTMs are designed to learn these temporal dependencies.

  Regular neural networks treat each input independently.
  LSTMs have "memory" — they remember what happened in previous time steps.

HOW LSTM WORKS (simplified):
  At each time step t, the LSTM receives:
    - Current weather: x_t (temperature, humidity, wind, etc.)
    - Previous hidden state: h_{t-1} (memory of past weather)

  It produces:
    - New hidden state: h_t (updated memory)
    - Output: used for prediction

  After processing all 14 days, the final hidden state
  captures the "accumulated fire danger" from recent weather.

INPUT:
  Weather time series: shape (batch, 14_days, 8_features)
  8 features = temp_max, temp_min, precip, wind, humidity,
               fire_danger_score, consecutive_dry_days, vpd_kpa

OUTPUT:
  Weather feature vector: shape (batch, 64)
  64-dimensional representation of temporal fire risk
  Combined with CNN output in hybrid_model.py

ARCHITECTURE:
  Input (batch, 14, 8)
      ↓
  LSTM(hidden=64, layers=2, dropout=0.2)
      ↓
  Take last hidden state → (batch, 64)
      ↓
  Linear(64 → 64) + ReLU
      ↓
  Output: (batch, 64)
"""

import torch
import torch.nn as nn
import numpy as np


class WeatherLSTM(nn.Module):
    """
    LSTM for processing weather time series leading up to a satellite image.

    Parameters:
        input_size   : Number of weather features per day (default 8)
        hidden_size  : LSTM hidden state size (default 64)
        num_layers   : Number of stacked LSTM layers (default 2)
        feature_dim  : Output feature vector size (default 64)
        dropout_rate : Dropout between LSTM layers (default 0.2)

    Usage:
        model = WeatherLSTM(input_size=8, hidden_size=64)
        weather = torch.randn(32, 14, 8)   # batch=32, days=14, features=8
        features = model(weather)           # shape: (32, 64)
    """

    def __init__(
        self,
        input_size  : int = 8,
        hidden_size : int = 64,
        num_layers  : int = 2,
        feature_dim : int = 64,
        dropout_rate: float = 0.2
    ):
        super(WeatherLSTM, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.feature_dim = feature_dim

        # ── LSTM layers
        # batch_first=True means input shape is (batch, seq_len, features)
        # instead of PyTorch default (seq_len, batch, features)
        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout_rate if num_layers > 1 else 0,
            bidirectional= False   # Unidirectional: past → future
        )

        # ── Layer normalization (stabilizes LSTM training)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # ── Final projection layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.

        Parameters:
            x : Weather time series (batch, seq_len, input_size)
                Example: (32, 14, 8) = 32 samples, 14 days, 8 features

        Returns:
            Feature vector (batch, feature_dim)
        """
        batch_size = x.size(0)

        # Initialize hidden state and cell state to zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Run through LSTM
        # lstm_out shape: (batch, seq_len, hidden_size)
        # We only need the FINAL time step output
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # Take the last time step output
        # lstm_out[:, -1, :] = output at day 14 (most recent day)
        # This captures accumulated weather history
        last_output = lstm_out[:, -1, :]   # (batch, hidden_size)

        # Apply layer normalization
        last_output = self.layer_norm(last_output)

        # Final projection
        features = self.fc(last_output)   # (batch, feature_dim)

        return features

    def get_output_dim(self) -> int:
        return self.feature_dim


# ─── ATTENTION-WEIGHTED LSTM (OPTIONAL UPGRADE) ───────────────────────────────

class WeatherLSTMWithAttention(nn.Module):
    """
    Enhanced LSTM with attention mechanism.

    WHY ATTENTION?
      Not all 14 days are equally important.
      The day before a fire is more important than 2 weeks ago.
      Attention learns to weight recent dangerous days higher.

    This is an upgrade over the base WeatherLSTM.
    Include this in your GSoC proposal as a planned improvement.
    """

    def __init__(self, input_size: int = 8, hidden_size: int = 64,
                 num_layers: int = 2, feature_dim: int = 64):
        super(WeatherLSTMWithAttention, self).__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.2 if num_layers > 1 else 0
        )

        # Attention scoring network
        # Takes each day's LSTM output → produces attention weight
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)

        # Get all time step outputs
        lstm_out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden)

        # Compute attention weights
        attn_scores  = self.attention(lstm_out)        # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # normalize over time

        # Weighted sum of all time steps
        # Days with higher attention weight contribute more
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)
        context = self.layer_norm(context)

        return torch.relu(self.fc(context))

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights for interpretability — which days mattered most."""
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        attn_scores  = self.attention(lstm_out)
        return torch.softmax(attn_scores, dim=1).squeeze(-1)  # (batch, seq_len)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Weather LSTM — Alaska Wildfire Pipeline ===\n")

    batch_size   = 8
    seq_len      = 14    # 14 days lookback
    n_features   = 8     # weather features

    print(f"Input: batch={batch_size}, days={seq_len}, features={n_features}")

    # ── Test base LSTM
    print("\n--- Base LSTM ---")
    model = WeatherLSTM(input_size=n_features, hidden_size=64, feature_dim=64)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    dummy_weather = torch.randn(batch_size, seq_len, n_features)
    model.eval()
    with torch.no_grad():
        features = model(dummy_weather)

    print(f"Input shape  : {dummy_weather.shape}")
    print(f"Output shape : {features.shape}")
    assert features.shape == (batch_size, 64)
    print("[PASS] Base LSTM working!")

    # ── Test attention LSTM
    print("\n--- LSTM with Attention ---")
    attn_model = WeatherLSTMWithAttention(input_size=n_features, hidden_size=64)

    attn_model.eval()
    with torch.no_grad():
        attn_features = attn_model(dummy_weather)
        attn_weights  = attn_model.get_attention_weights(dummy_weather)

    print(f"Output shape          : {attn_features.shape}")
    print(f"Attention weights shape: {attn_weights.shape}  ← one weight per day")
    print(f"Attention weights sum  : {attn_weights[0].sum().item():.4f}  ← should be ~1.0")
    print("[PASS] Attention LSTM working!")

    print(f"\n[DONE] Both LSTM variants ready for hybrid model.")