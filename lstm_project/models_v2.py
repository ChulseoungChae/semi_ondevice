"""
1D CNN and LSTM Models for PVD PWPDS.Data Prediction (V2)
- PVD1DCNNModel: Conv1d-based model
- PVDLSTMModelV2: Bidirectional LSTM with Attention
- 8 input features, no PWPDS.Data in input
"""

import torch
import torch.nn as nn


class PVD1DCNNModel(nn.Module):
    """1D CNN model for PVD time series prediction.

    Architecture:
    (batch, 10, 8) -> permute -> (batch, 8, 10)
    Conv1d(8->64, k=3) + BN + ReLU
    Conv1d(64->128, k=3) + BN + ReLU
    Conv1d(128->128, k=3) + BN + ReLU
    AdaptiveAvgPool1d(1) -> (batch, 128)
    FC(128->64) + ReLU + Dropout
    FC(64->1)
    """

    def __init__(self, n_features: int, n_outputs: int = 1, dropout: float = 0.2):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
        x = self.conv_layers(x)  # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)
        x = self.fc(x)  # (batch, n_outputs)
        return x


class PVDLSTMModelV2(nn.Module):
    """Bidirectional LSTM with Attention for PVD time series prediction.

    Architecture:
    (batch, 10, 8) -> InputProj(8->128)
    BiLSTM(128, 2 layers) -> Attention -> (batch, 256)
    FC(256->128->64->1)
    """

    def __init__(self, n_features: int, n_outputs: int = 1,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

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

        output = self.output_layer(context)  # (batch, n_outputs)
        return output


if __name__ == '__main__':
    n_feat = 8
    batch_size = 16

    print("=== PVD1DCNNModel ===")
    cnn = PVD1DCNNModel(n_features=n_feat)
    x = torch.randn(batch_size, 10, n_feat)
    y = cnn(x)
    params = sum(p.numel() for p in cnn.parameters())
    print(f"  Input: {x.shape} -> Output: {y.shape}, Params: {params:,}")

    print("\n=== PVDLSTMModelV2 ===")
    lstm = PVDLSTMModelV2(n_features=n_feat)
    y = lstm(x)
    params = sum(p.numel() for p in lstm.parameters())
    print(f"  Input: {x.shape} -> Output: {y.shape}, Params: {params:,}")
