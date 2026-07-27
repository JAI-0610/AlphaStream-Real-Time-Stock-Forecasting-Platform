import torch
import torch.nn as nn
from typing import List

class CNNLSTMForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        conv_filters: int = 64,
        kernel_size: int = 3,
        lstm_hidden_dim: int = 128,
        dropout: float = 0.2,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # Conv1d expects shape: [batch, channels, length]
        # In our case: channels=input_dim, length=sequence_length
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2) # Reduces sequence length by half
        
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_dim, self.num_quantiles)

    def forward(self, x: torch.Tensor, static_cov: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, sequence_length, input_dim]
            static_cov: Unused, kept for interface match
        """
        # Rearrange to [batch_size, input_dim, sequence_length]
        x_conv = x.transpose(1, 2)
        
        # Conv -> Relu -> Pool
        conv_out = self.conv1d(x_conv)
        conv_out = self.relu(conv_out)
        conv_out = self.pool(conv_out) # Shape: [batch_size, conv_filters, sequence_length / 2]
        
        # Rearrange to [batch_size, sequence_length / 2, conv_filters] for LSTM
        lstm_input = conv_out.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(lstm_input)
        
        # Pull final sequence representation
        last_timestep = lstm_out[:, -1, :] # [batch_size, lstm_hidden_dim]
        
        # Project and output
        out = self.dropout(last_timestep)
        out = self.fc(out) # [batch_size, num_quantiles]
        return out
