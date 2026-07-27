import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, List
from torch.utils.data import DataLoader
from models.loss import QuantileLoss
from models.lstm import LSTMForecaster
from models.gru import GRUForecaster
from models.cnn_lstm import CNNLSTMForecaster
from models.transformer_tft import TemporalFusionTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_trainer")

class ModelTrainer:
    def __init__(self, model_name: str, input_dim: int, config_path: str = "configs/config.yaml"):
        self.model_name = model_name
        self.input_dim = input_dim
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.epochs: int = self.config["training"]["epochs"]
        self.lr: float = self.config["training"]["learning_rate"]
        self.weight_decay: float = self.config["training"]["weight_decay"]
        self.patience: int = self.config["training"]["early_stopping_patience"]
        self.save_dir: str = self.config["training"]["save_dir"]
        self.device_config: str = self.config["training"]["device"]
        
        # Auto-detect device
        if self.device_config == "auto" or self.device_config == "cpu":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.device_config)
            
        logger.info(f"Using device: {self.device}")
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Quantiles config
        self.quantiles = self.config["model"]["tft"]["quantiles"]
        
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Loss & Optimizer
        self.criterion = QuantileLoss(quantiles=self.quantiles)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode="min", 
            factor=0.5, 
            patience=3
        )

    def _build_model(self) -> nn.Module:
        """Instantiates the selected model model based on config."""
        logger.info(f"Building model: {self.model_name}")
        if self.model_name == "lstm":
            cfg = self.config["model"]["lstm"]
            return LSTMForecaster(
                input_dim=self.input_dim,
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                quantiles=self.quantiles
            )
        elif self.model_name == "gru":
            cfg = self.config["model"]["gru"]
            return GRUForecaster(
                input_dim=self.input_dim,
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                quantiles=self.quantiles
            )
        elif self.model_name == "cnn_lstm":
            cfg = self.config["model"]["cnn_lstm"]
            return CNNLSTMForecaster(
                input_dim=self.input_dim,
                conv_filters=cfg["conv_filters"],
                kernel_size=cfg["kernel_size"],
                lstm_hidden_dim=cfg["lstm_hidden_dim"],
                dropout=cfg["dropout"],
                quantiles=self.quantiles
            )
        elif self.model_name == "tft":
            cfg = self.config["model"]["tft"]
            return TemporalFusionTransformer(
                input_dim=self.input_dim,
                static_num_sectors=5, # Static mapping supports up to 5 sectors
                hidden_dim=cfg["hidden_dim"],
                num_heads=cfg["num_heads"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                quantiles=self.quantiles
            )
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        epoch_loss = 0.0
        
        for batch in dataloader:
            x = batch["x"].to(self.device)
            static_cov = batch["static_cov"].to(self.device)
            target = batch["target_return"].to(self.device)
            
            self.optimizer.zero_grad()
            
            preds = self.model(x, static_cov)
            loss = self.criterion(preds, target)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in Transformers
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
            
        return epoch_loss / len(dataloader.dataset)

    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(self.device)
                static_cov = batch["static_cov"].to(self.device)
                target = batch["target_return"].to(self.device)
                
                preds = self.model(x, static_cov)
                loss = self.criterion(preds, target)
                val_loss += loss.item() * x.size(0)
                
        return val_loss / len(dataloader.dataset)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        logger.info("Starting model training...")
        history = {"train_loss": [], "val_loss": []}
        
        best_val_loss = float("inf")
        patience_counter = 0
        checkpoint_path = os.path.join(self.save_dir, f"best_{self.model_name}.pth")
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            logger.info(f"Epoch {epoch:02d}/{self.epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Early stopping & Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model weights and configuration metadata
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "input_dim": self.input_dim,
                    "model_name": self.model_name
                }, checkpoint_path)
                logger.info(f"Saved best checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.warning(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                    break
                    
        return history

    def load_best_model(self):
        """Loads the saved best checkpoint weights into the model."""
        checkpoint_path = os.path.join(self.save_dir, f"best_{self.model_name}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Successfully loaded best model from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")


if __name__ == "__main__":
    import yaml
    from data_ingestion.historical_downloader import HistoricalDownloader
    from data_validation.validator import DataValidator
    from feature_engineering.pipeline import FeaturePipeline
    from datasets.market_dataset import create_dataloaders

    # 1. Ingest
    logger.info("Initializing data pipeline...")
    downloader = HistoricalDownloader()
    data_dict = downloader.run_all()
    
    # 2. Validate
    logger.info("Validating historical data...")
    validated_dict = {}
    for ticker, df in data_dict.items():
        validated_dict[ticker] = DataValidator.clean_and_validate(df)
        
    # 3. Features
    logger.info("Extracting features...")
    pipeline = FeaturePipeline()
    processed_dfs = pipeline.build_multivariate_dataset(validated_dict, is_training=True)
    
    # 4. Dataloaders
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    train_loader, val_loader, feature_cols = create_dataloaders(
        processed_dfs,
        sequence_length=config["data"]["sequence_length"],
        batch_size=config["training"]["batch_size"],
        train_pct=0.8,
        has_targets=True
    )
    
    logger.info(f"Loaded {len(feature_cols)} features: {feature_cols}")
    
    # 5. Train all models sequentially to enable multi-model ensemble forecasting
    models_to_train = ["tft", "lstm", "gru", "cnn_lstm"]
    logger.info(f"Beginning training sequence for models: {models_to_train}")
    
    for m_name in models_to_train:
        logger.info(f"==================================================")
        logger.info(f"Starting training for model: {m_name}")
        logger.info(f"==================================================")
        try:
            trainer = ModelTrainer(
                model_name=m_name,
                input_dim=len(feature_cols)
            )
            history = trainer.fit(train_loader, val_loader)
            logger.info(f"Completed training for model: {m_name}")
        except Exception as e:
            logger.error(f"Error training model {m_name}: {e}")
            
    logger.info("All model training sequences completed!")
