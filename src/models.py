from typing import Tuple

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """
    Minimal LSTM-based sequence-to-sequence forecaster.

    Input:  [B, L, D]
    Output: [B, H, D]
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 1,
        horizon: int = 16,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, L, D].

        Returns
        -------
        torch.Tensor
            Forecast of shape [B, H, D].
        """
        out, _ = self.lstm(x)
        # Use the last time-step representation.
        last = out[:, -1, :]  # [B, hidden_dim]
        y = self.head(last)  # [B, H * D]
        return y.view(-1, self.horizon, self.output_dim)


class TransformerForecaster(nn.Module):
    """
    Minimal Transformer encoder-based forecaster.

    Input:  [B, L, D]
    Output: [B, H, D]
    """

    def __init__(
        self,
        input_dim: int = 3,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        max_len: int = 512,
        horizon: int = 16,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.max_len = max_len

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, max_len, model_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, horizon * output_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, L, D].

        Returns
        -------
        torch.Tensor
            Forecast of shape [B, H, D].
        """
        B, L, _ = x.shape
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} exceeds max_len={self.max_len}")

        h = self.input_proj(x)  # [B, L, model_dim]
        pos_emb = self.positional_embedding[:, :L, :]
        h = h + pos_emb

        enc = self.encoder(h)  # [B, L, model_dim]
        pooled = enc.mean(dim=1)  # simple mean pooling over time

        y = self.head(pooled)  # [B, H * D]
        return y.view(B, self.horizon, self.output_dim)

