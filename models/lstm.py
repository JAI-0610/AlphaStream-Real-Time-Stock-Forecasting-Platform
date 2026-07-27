import torch
import torch.nn as nn
from typing import List

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output project head: outputs the predictions for the specified quantiles
        self.fc = nn.Linear(hidden_dim, self.num_quantiles)

    def forward(self, x: torch.Tensor, static_cov: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, sequence_length, input_dim]
            static_cov: Optional static context (unused in baseline, used for API compatibility)
        Returns:
            preds: Tensor of shape [batch_size, num_quantiles]
        """
        # lstm_out: [batch_size, sequence_length, hidden_dim]
        # hn, cn: [num_layers, batch_size, hidden_dim]
        lstm_out, _ = self.lstm(x)
        
        # Take the final hidden state representation of the sequence
        last_timestep = lstm_out[:, -1, :] # [batch_size, hidden_dim]
        
        # Project to quantile outputs
        out = self.fc(last_timestep) # [batch_size, num_quantiles]
        return out
