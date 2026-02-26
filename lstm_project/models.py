"""
LSTM Models for PVD PWPDS.Data Prediction
- 4 model variants (A/B/C/D) with different input features
- Shared LSTM architecture, different input dimensions
"""

import torch
import torch.nn as nn


class PVDLSTMModel(nn.Module):
    """LSTM model for PVD time series prediction.

    Architecture:
    - 2-layer Bidirectional LSTM with dropout
    - Attention mechanism over time steps
    - Fully connected output layers

    Input: (batch, seq_len=10, n_features)
    Output: (batch, n_outputs=1)
    """

    def __init__(self, n_features: int, n_outputs: int = 1,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_outputs),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)  # (batch, seq_len, hidden)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)

        # Output
        output = self.output_layer(context)  # (batch, n_outputs)
        return output


class PVDBoundsModel(nn.Module):
    """Model for predicting PWPDS.Data upper/lower bounds.

    Takes last 10s of EN4.Power and SBRF5.SetPower as input,
    outputs upper and lower bounds.

    Architecture: Simple MLP (DC/RF settings are relatively stable)
    """

    def __init__(self, input_size: int = 2, seq_len: int = 10, hidden_size: int = 64):
        super().__init__()

        # Process the sequence with 1D conv + pooling
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (batch, hidden, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 2),  # lower_bound, upper_bound
        )

    def forward(self, x):
        # x: (batch, seq_len, 2) -> permute to (batch, 2, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)  # (batch, hidden)
        bounds = self.fc(x)  # (batch, 2)
        # Ensure lower < upper: use raw output as (center, half_range)
        center = bounds[:, 0:1]
        half_range = torch.abs(bounds[:, 1:2]) + 1e-6
        lower = center - half_range
        upper = center + half_range
        return torch.cat([lower, upper], dim=1)


def get_model(model_key: str, n_features: int, n_outputs: int = 1, **kwargs) -> PVDLSTMModel:
    """Create LSTM model for given configuration."""
    return PVDLSTMModel(n_features=n_features, n_outputs=n_outputs, **kwargs)


if __name__ == '__main__':
    # Test all model configurations
    from data_loader import MODEL_CONFIGS

    for key, config in MODEL_CONFIGS.items():
        n_feat = len(config['input_cols'])
        n_out = len(config['output_cols'])
        model = get_model(key, n_feat, n_out)
        x = torch.randn(8, 10, n_feat)
        y = model(x)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model {key}: input={n_feat}, output={n_out}, "
              f"out_shape={y.shape}, params={total_params:,}")

    # Test bounds model
    bounds_model = PVDBoundsModel()
    x = torch.randn(8, 10, 2)
    bounds = bounds_model(x)
    print(f"\nBounds model: input=(8,10,2), output={bounds.shape}")
    print(f"  Lower: {bounds[:2, 0].detach()}, Upper: {bounds[:2, 1].detach()}")
