import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

# Sector encoding mapping
SECTOR_MAPPING = {
    # Technology (0)
    "AAPL": 0, "MSFT": 0, "GOOGL": 0, "META": 0, "NVDA": 0, "ADBE": 0, "CRM": 0, "AMD": 0, "INTC": 0, "ORCL": 0, "QCOM": 0, "AVGO": 0,
    # Comm. Services (1)
    "NFLX": 1,
    # Consumer Discretionary / Staples / Industrials (2)
    "AMZN": 2, "TSLA": 2, "KO": 2, "PEP": 2, "WMT": 2, "PG": 2, "BA": 2, "CAT": 2, "GE": 2,
    # ETFs (3)
    "SPY": 3, "QQQ": 3, "DIA": 3,
    # Financials (4)
    "XLF": 4, "JPM": 4, "BAC": 4, "GS": 4, "MS": 4, "V": 4, "MA": 4, "BRK-B": 4,
    # Healthcare / Energy (mapping to 0 or any other valid index)
    "JNJ": 0, "PFE": 0, "UNH": 0, "ABBV": 0, "MRK": 0,
    "XOM": 0, "CVX": 0, "COP": 0
}

class MarketSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sequence_length: int = 30, has_targets: bool = True):
        self.df = df.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.has_targets = has_targets
        
        # Get ticker sector ID
        ticker = self.df["ticker"].iloc[0] if "ticker" in self.df.columns else "AAPL"
        self.sector_id = SECTOR_MAPPING.get(ticker, 0)
        
        # Extract features (drop metadata and target columns)
        metadata_cols = ["timestamp", "ticker", "open", "high", "low", "close", "volume", "target_return", "target_direction"]
        self.feature_cols = [c for c in self.df.columns if c not in metadata_cols]
        
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        
        if self.has_targets:
            self.target_returns = self.df["target_return"].values.astype(np.float32)
            self.target_directions = self.df["target_direction"].values.astype(np.float32)
            
        # Number of samples we can build
        self.num_samples = len(self.df) - self.sequence_length + 1
        
        if self.num_samples <= 0:
            raise ValueError(f"Data length ({len(self.df)}) is less than sequence length ({self.sequence_length})")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sequence slicing
        start_idx = idx
        end_idx = idx + self.sequence_length
        
        x_seq = self.features[start_idx:end_idx] # Shape: [sequence_length, num_features]
        
        # We output a dictionary of inputs
        item = {
            "x": torch.tensor(x_seq, dtype=torch.float32),
            "static_cov": torch.tensor(self.sector_id, dtype=torch.long)
        }
        
        if self.has_targets:
            # The target is corresponding to the last step of the sequence (end_idx - 1)
            target_idx = end_idx - 1
            item["target_return"] = torch.tensor(self.target_returns[target_idx], dtype=torch.float32)
            item["target_direction"] = torch.tensor(self.target_directions[target_idx], dtype=torch.float32)
            
        return item

def create_dataloaders(
    processed_dfs: Dict[str, pd.DataFrame],
    sequence_length: int = 30,
    batch_size: int = 64,
    train_pct: float = 0.8,
    has_targets: bool = True
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Splits processed dataframes sequentially (no random shuffling to avoid temporal leaks)
    and returns PyTorch DataLoaders.
    """
    train_list = []
    val_list = []
    feature_cols = None
    
    for ticker, df in processed_dfs.items():
        # Clean targets (ensure no NaNs in targets for dataset)
        if has_targets:
            df_clean = df.dropna(subset=["target_return", "target_direction"]).reset_index(drop=True)
        else:
            df_clean = df.copy()
            
        n = len(df_clean)
        split_idx = int(n * train_pct)
        
        train_df = df_clean.iloc[:split_idx].reset_index(drop=True)
        val_df = df_clean.iloc[split_idx:].reset_index(drop=True)
        
        # Build datasets
        if len(train_df) >= sequence_length:
            train_ds = MarketSequenceDataset(train_df, sequence_length, has_targets)
            train_list.append(train_ds)
            if feature_cols is None:
                feature_cols = train_ds.feature_cols
                
        if len(val_df) >= sequence_length:
            val_ds = MarketSequenceDataset(val_df, sequence_length, has_targets)
            val_list.append(val_ds)
            
    # Combine datasets
    if not train_list or not val_list:
        raise ValueError("Insufficient data to build training/validation dataloaders.")
        
    combined_train = torch.utils.data.ConcatDataset(train_list)
    combined_val = torch.utils.data.ConcatDataset(val_list)
    
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(combined_val, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, feature_cols
