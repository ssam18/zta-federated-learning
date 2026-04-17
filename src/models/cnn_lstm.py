"""
CNN-LSTM model for IIoT network intrusion detection.

Architecture: 1-D convolutional feature extractor followed by a bidirectional LSTM
for temporal dependency modelling.  The model is designed to operate on fixed-length
feature windows extracted from network packet / flow records.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMClassifier(nn.Module):
    """
    One-dimensional CNN followed by a stacked LSTM for multi-class traffic classification.

    Parameters
    ----------
    n_features : int
        Number of input features per time-step.
    n_classes : int
        Number of output classes.
    seq_len : int
        Number of time-steps in each input window.
    cnn_filters : int
        Number of convolutional filters in the first two conv blocks.
    cnn_kernel : int
        Kernel size for all convolutional layers.
    lstm_hidden : int
        Hidden size of each LSTM layer.
    lstm_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability applied between LSTM layers and before the FC head.
    bidirectional : bool
        Whether to use a bidirectional LSTM.
    """

    def __init__(
        self,
        n_features: int = 40,
        n_classes: int = 15,
        seq_len: int = 1,
        cnn_filters: int = 64,
        cnn_kernel: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional

        # --- Convolutional blocks -----------------------------------------------
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel,
            padding=cnn_kernel // 2,
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters)

        self.conv2 = nn.Conv1d(
            in_channels=cnn_filters,
            out_channels=cnn_filters * 2,
            kernel_size=cnn_kernel,
            padding=cnn_kernel // 2,
        )
        self.bn2 = nn.BatchNorm1d(cnn_filters * 2)

        self.conv3 = nn.Conv1d(
            in_channels=cnn_filters * 2,
            out_channels=cnn_filters * 2,
            kernel_size=cnn_kernel,
            padding=cnn_kernel // 2,
        )
        self.bn3 = nn.BatchNorm1d(cnn_filters * 2)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.cnn_dropout = nn.Dropout(p=dropout)

        # After pooling the temporal dimension is halved; compute adaptive output
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=4)

        cnn_out_dim = cnn_filters * 2 * 4  # channels × pooled length

        # --- LSTM block ----------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=cnn_filters * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out = lstm_hidden * (2 if bidirectional else 1)

        # --- Fully-connected head ------------------------------------------------
        self.fc_dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(lstm_out, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_features)`` for single time-step inference, or
            ``(batch, n_features, seq_len)`` for windowed input.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(batch, n_classes)``.
        """
        # Ensure shape is (batch, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, F, 1)

        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        # MaxPool only when sequence length allows it (skip for single-step input)
        if x.size(-1) >= 2:
            x = self.pool(x)
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cnn_dropout(x)

        # Adaptive pool → fixed (B, C, 4)
        x = self.adaptive_pool(x)  # (B, cnn_filters*2, 4)

        # Rearrange for LSTM: (B, seq=4, features=cnn_filters*2)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)          # (B, 4, lstm_out)
        x = lstm_out[:, -1, :]              # last time-step (B, lstm_out)

        # FC head
        x = self.fc_dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    # ------------------------------------------------------------------
    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate layer activations (before the final FC)."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if x.size(-1) >= 2:
            x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cnn_dropout(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc_dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# ---------------------------------------------------------------------------
# Quantisation utility
# ---------------------------------------------------------------------------

def quantize_model(model: CNNLSTMClassifier) -> torch.nn.Module:
    """
    Apply dynamic 8-bit quantisation to a trained CNN-LSTM model.

    Dynamic quantisation converts weights to INT8 and performs activations
    in INT8 at run time, reducing memory footprint and inference latency on
    CPU deployments (e.g., edge devices).

    Parameters
    ----------
    model : CNNLSTMClassifier
        A trained model instance.

    Returns
    -------
    torch.nn.Module
        The quantised model.  Note that quantised models must remain on CPU.
    """
    model.cpu().eval()
    quantised = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )
    return quantised


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def model_summary(model: CNNLSTMClassifier, input_shape: tuple[int, ...] | None = None) -> str:
    """
    Return a formatted string summarising the model architecture.

    Parameters
    ----------
    model : CNNLSTMClassifier
    input_shape : tuple, optional
        ``(n_features,)`` or ``(n_features, seq_len)`` for a single sample.

    Returns
    -------
    str
        Human-readable summary.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("CNN-LSTM Intrusion Detection Classifier")
    lines.append("=" * 60)
    for name, module in model.named_modules():
        if name == "":
            continue
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        if n_params > 0:
            lines.append(f"  {name:<30s}  {str(module.__class__.__name__):<20s}  params={n_params:,d}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append("-" * 60)
    lines.append(f"  Total parameters     : {total:,d}")
    lines.append(f"  Trainable parameters : {trainable:,d}")
    lines.append("=" * 60)
    if input_shape is not None:
        lines.append(f"  Expected input shape : {input_shape}")
    return "\n".join(lines)
