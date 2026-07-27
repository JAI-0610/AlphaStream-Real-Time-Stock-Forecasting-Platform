import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) from TFT.
    Allows features to pass through unmodified if they are already optimal,
    using Gated Linear Units (GLU) and skip connections.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim * 2) # For GLU gating
        self.gate = nn.GLU(dim=-1)
        
        # Skip connection adjustment if dimensions differ
        self.skip_projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, D_in] or [B, D_in]
        h = F.elu(self.linear1(x))
        h = self.dropout(h)
        h = self.gate(self.linear2(h)) # Output matches output_dim
        
        skip = self.skip_projection(x)
        return self.layer_norm(h + skip)

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) from TFT.
    Computes importance scores for each feature, allowing the model to filter noise
    and output explainable feature weights.
    """
    def __init__(self, num_features: int, feature_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Project each input feature to hidden_dim
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(feature_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])
        
        # Network to compute variable selection weights
        self.weight_grn = GatedResidualNetwork(
            input_dim=num_features * feature_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            dropout=dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [B, T, num_features]
        Returns:
            vsn_out: Projected and combined features [B, T, hidden_dim]
            weights: Feature selection weights [B, T, num_features]
        """
        B, T, F_dim = x.shape
        assert F_dim == self.num_features, f"Input dim {F_dim} does not match feature count {self.num_features}"
        
        # Compute selection weights
        # Flatten sequence/features to feed to weights GRN
        # We process each timestep independently
        flat_x = x.unsqueeze(-1) # [B, T, F, 1] (each feature has dim 1)
        
        # Prepare inputs for single variable networks
        var_outputs = []
        for i in range(self.num_features):
            var_feat = x[..., i:i+1] # [B, T, 1]
            var_out = self.single_variable_grns[i](var_feat) # [B, T, hidden_dim]
            var_outputs.append(var_out)
        
        # Stack processed variables: [B, T, num_features, hidden_dim]
        stacked_var_outputs = torch.stack(var_outputs, dim=-2)
        
        # Get weight scores using the raw variables concatenated
        raw_concat = x # [B, T, num_features]
        weight_scores = self.weight_grn(raw_concat) # [B, T, num_features]
        weights = self.softmax(weight_scores) # [B, T, num_features]
        
        # Weighted sum: multiply weights with processed variables
        # weights: [B, T, num_features, 1]
        vsn_out = torch.sum(weights.unsqueeze(-1) * stacked_var_outputs, dim=-2) # [B, T, hidden_dim]
        
        return vsn_out, weights

class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_num_sectors: int = 5,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # Static covariate embedding
        self.static_embedding = nn.Embedding(static_num_sectors, hidden_dim)
        self.static_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # Feature Selection Network
        self.vsn = VariableSelectionNetwork(
            num_features=input_dim,
            feature_dim=1,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Temporal self-attention layer (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Context-based post-attention GRN
        self.post_attn_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # Multi-quantile output head
        self.quantile_head = nn.Linear(hidden_dim, self.num_quantiles)
        
        # Caching for explainability reports
        self.last_variable_weights = None
        self.last_attention_maps = None

    def forward(self, x: torch.Tensor, static_cov: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, input_dim]
            static_cov: Sector embedding indices [B] or [B, 1]
        Returns:
            preds: Quantile forecasts [B, num_quantiles]
        """
        B, T, F = x.shape
        
        # 1. Encode static covariate (e.g. Sector ID)
        if static_cov is None:
            static_cov = torch.zeros(B, dtype=torch.long, device=x.device)
        elif static_cov.ndim > 1:
            static_cov = static_cov.squeeze(-1)
            
        static_embed = self.static_embedding(static_cov) # [B, hidden_dim]
        static_ctx = self.static_grn(static_embed).unsqueeze(1) # [B, 1, hidden_dim]
        
        # 2. Variable Selection Network
        vsn_out, var_weights = self.vsn(x) # vsn_out: [B, T, hidden_dim], var_weights: [B, T, F]
        self.last_variable_weights = var_weights.detach() # Cache for explainability
        
        # Add static context to selection features
        vsn_out = vsn_out + static_ctx
        
        # 3. Temporal Multi-Head Attention
        # PyTorch Transformer does self-attention
        attn_out = self.transformer_encoder(vsn_out) # [B, T, hidden_dim]
        
        # 4. Post-Attention Gated Residual and Projection
        # Extract the representation of the final time step
        final_state = attn_out[:, -1, :] # [B, hidden_dim]
        final_state = self.post_attn_grn(final_state) # [B, hidden_dim]
        
        # Project to multiple quantiles
        out = self.quantile_head(final_state) # [B, num_quantiles]
        return out

    def get_variable_importances(self) -> torch.Tensor:
        """Returns the cached variable weights of shape [B, T, input_dim]."""
        return self.last_variable_weights
