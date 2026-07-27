import torch
import torch.nn as nn
from typing import List

class QuantileLoss(nn.Module):
    """Pinball Loss for multiple quantiles.
    Enables the model to learn prediction intervals (uncertainty bounds)
    instead of just point estimates.
    """
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Tensor of shape [batch_size, len(quantiles)] or [batch_size, horizon, len(quantiles)]
            target: Tensor of shape [batch_size] or [batch_size, horizon]
        """
        # Ensure target matches preds size except for the last dimension (quantiles)
        if target.ndim == 1:
            target = target.unsqueeze(-1)  # Shape: [batch_size, 1]
        elif target.ndim == 2 and preds.ndim == 3:
            target = target.unsqueeze(-1)  # Shape: [batch_size, horizon, 1]
            
        losses = []
        for i, q in enumerate(self.quantiles):
            error = target - preds[..., i:i+1]
            loss = torch.max((q - 1) * error, q * error)
            losses.append(loss.mean())
            
        return torch.stack(losses).mean()
