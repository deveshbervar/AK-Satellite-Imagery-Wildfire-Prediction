"""
cnn_feature_extractor.py
-------------------------
CNN model that extracts spatial features from Sentinel-2 satellite patches.

WHY CNN FOR SATELLITE IMAGERY?
  Satellite images are 2D spatial grids — CNNs are designed exactly for this.
  Each convolutional layer learns to detect spatial patterns like:
    - Layer 1: Edges, color boundaries
    - Layer 2: Vegetation patches, water bodies
    - Layer 3: Fire-prone forest types, terrain patterns

INPUT:
  Sentinel-2 patches: shape (batch, 6_bands, 64, 64)
  6 bands = B2(Blue), B3(Green), B4(Red), B8(NIR), B11(SWIR1), B12(SWIR2)

OUTPUT:
  Feature vector: shape (batch, 128)
  128-dimensional representation of spatial fire risk features
  This gets passed to the hybrid model which combines it with LSTM output.

ARCHITECTURE:
  Input (6, 64, 64)
      ↓
  Conv2D(32 filters, 3x3) + BatchNorm + ReLU  → (32, 64, 64)
      ↓
  MaxPool(2x2)                                 → (32, 32, 32)
      ↓
  Conv2D(64 filters, 3x3) + BatchNorm + ReLU  → (64, 32, 32)
      ↓
  MaxPool(2x2)                                 → (64, 16, 16)
      ↓
  Conv2D(128 filters, 3x3) + BatchNorm + ReLU → (128, 16, 16)
      ↓
  GlobalAveragePooling                         → (128,)
      ↓
  Output feature vector: (128,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """
    A single convolutional block: Conv2D → BatchNorm → ReLU

    WHY BATCHNORM?
      Normalizes activations between layers.
      Prevents exploding/vanishing gradients.
      Makes training faster and more stable.

    WHY RELU?
      Simple non-linearity. Turns off negative activations.
      Prevents the network from being purely linear.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            padding      = kernel_size // 2,   # 'same' padding keeps spatial size
            bias         = False                # BatchNorm handles bias
        )
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SentinelCNN(nn.Module):
    """
    CNN for extracting spatial fire-risk features from Sentinel-2 patches.

    Parameters:
        in_channels    : Number of input bands (default 6: B2,B3,B4,B8,B11,B12)
        feature_dim    : Size of output feature vector (default 128)
        dropout_rate   : Dropout probability for regularization (default 0.3)

    Usage:
        model = SentinelCNN(in_channels=6, feature_dim=128)
        patch = torch.randn(32, 6, 64, 64)   # batch of 32 patches
        features = model(patch)               # shape: (32, 128)
    """

    def __init__(self, in_channels: int = 6, feature_dim: int = 128, dropout_rate: float = 0.3):
        super(SentinelCNN, self).__init__()

        self.in_channels  = in_channels
        self.feature_dim  = feature_dim

        # ── Convolutional backbone
        self.block1 = ConvBlock(in_channels, 32)    # 6  → 32 channels
        self.pool1  = nn.MaxPool2d(2, 2)             # 64x64 → 32x32

        self.block2 = ConvBlock(32, 64)              # 32 → 64 channels
        self.pool2  = nn.MaxPool2d(2, 2)             # 32x32 → 16x16

        self.block3 = ConvBlock(64, 128)             # 64 → 128 channels
        self.pool3  = nn.MaxPool2d(2, 2)             # 16x16 → 8x8

        self.block4 = ConvBlock(128, feature_dim)    # 128 → feature_dim

        # ── Global Average Pooling
        # Instead of flattening (which creates huge vectors),
        # GAP averages each feature map into a single number.
        # Input:  (batch, feature_dim, 8, 8) → Output: (batch, feature_dim)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # ── Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.

        Parameters:
            x : Tensor of shape (batch, in_channels, H, W)

        Returns:
            Feature vector of shape (batch, feature_dim)
        """
        # Convolutional blocks with pooling
        x = self.pool1(self.block1(x))   # (B, 32, 32, 32)
        x = self.pool2(self.block2(x))   # (B, 64, 16, 16)
        x = self.pool3(self.block3(x))   # (B, 128, 8, 8)
        x = self.block4(x)               # (B, feature_dim, 8, 8)

        # Global average pooling → (B, feature_dim, 1, 1)
        x = self.global_avg_pool(x)

        # Flatten → (B, feature_dim)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        return x

    def get_output_dim(self) -> int:
        return self.feature_dim


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== CNN Feature Extractor — Alaska Wildfire Pipeline ===\n")

    # Test with synthetic satellite patches
    batch_size  = 8
    n_bands     = 6
    patch_size  = 64

    print(f"Input: batch={batch_size}, bands={n_bands}, size={patch_size}x{patch_size}")

    # Create model
    model = SentinelCNN(in_channels=n_bands, feature_dim=128)
    print(f"\nModel architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Test forward pass
    dummy_input = torch.randn(batch_size, n_bands, patch_size, patch_size)
    model.eval()
    with torch.no_grad():
        features = model(dummy_input)

    print(f"\nForward pass test:")
    print(f"  Input shape  : {dummy_input.shape}")
    print(f"  Output shape : {features.shape}")
    print(f"  Expected     : ({batch_size}, 128)")
    assert features.shape == (batch_size, 128), "Shape mismatch!"
    print(f"\n[PASS] CNN feature extractor working correctly!")