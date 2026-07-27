import optuna
import logging
import yaml
import torch
from typing import Dict, Any
from training.trainer import ModelTrainer
from datasets.market_dataset import create_dataloaders
from data_ingestion.historical_downloader import HistoricalDownloader
from data_validation.validator import DataValidator
from feature_engineering.pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hyperparameter_tuning")

def objective(trial: optuna.Trial) -> float:
    # Load base configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Hyperparameters to search
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    # Temporarily override config parameters
    config["training"]["learning_rate"] = lr
    config["training"]["weight_decay"] = weight_decay
    config["training"]["epochs"] = 5  # Keep it small for fast tuning trials
    config["model"]["tft"]["hidden_dim"] = hidden_dim
    config["model"]["tft"]["dropout"] = dropout
    
    # Save temporary trial config
    trial_config_path = "configs/config_trial.yaml"
    with open(trial_config_path, "w") as f:
        yaml.safe_dump(config, f)
        
    # Load dataset
    downloader = HistoricalDownloader(config_path=trial_config_path)
    data_dict = downloader.run_all()
    
    validated_dict = {}
    for ticker, df in data_dict.items():
        validated_dict[ticker] = DataValidator.clean_and_validate(df)
        
    pipeline = FeaturePipeline(config_path=trial_config_path)
    processed_dfs = pipeline.build_multivariate_dataset(validated_dict, is_training=True)
    
    train_loader, val_loader, feature_cols = create_dataloaders(
        processed_dfs,
        sequence_length=config["data"]["sequence_length"],
        batch_size=config["training"]["batch_size"],
        train_pct=0.8,
        has_targets=True
    )
    
    # Initialize trainer and train
    trainer = ModelTrainer(
        model_name="tft",
        input_dim=len(feature_cols),
        config_path=trial_config_path
    )
    
    history = trainer.fit(train_loader, val_loader)
    
    # Return best validation loss achieved during trial
    val_loss = min(history["val_loss"])
    return val_loss

def run_tuning(n_trials: int = 5):
    logger.info("Initializing Optuna study for hyperparameter optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Tuning study completed!")
    logger.info(f"Best Trial value: {study.best_trial.value:.6f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Save best parameters back to main configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    config["training"]["learning_rate"] = study.best_params["learning_rate"]
    config["training"]["weight_decay"] = study.best_params["weight_decay"]
    config["model"]["tft"]["hidden_dim"] = study.best_params["hidden_dim"]
    config["model"]["tft"]["dropout"] = study.best_params["dropout"]
    
    with open("configs/config.yaml", "w") as f:
        yaml.safe_dump(config, f)
        
    logger.info("Saved optimized hyperparameters to configs/config.yaml")

if __name__ == "__main__":
    run_tuning(n_trials=3)
