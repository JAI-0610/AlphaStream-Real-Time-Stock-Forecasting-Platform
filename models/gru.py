import torch
import torch.nn as nn
from typing import List

class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_dim, self.num_quantiles)

    def forward(self, x: torch.Tensor, static_cov: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, sequence_length, input_dim]
            static_cov: Unused, kept for interface match
        """
        gru_out, _ = self.gru(x)
        last_timestep = gru_out[:, -1, :] # [batch_size, hidden_dim]
        out = self.fc(last_timestep) # [batch_size, num_quantiles]
        return out
