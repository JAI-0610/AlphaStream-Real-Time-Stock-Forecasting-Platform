import logging
import pandas as pd
import numpy as np
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data_validator")

class DataValidationError(Exception):
    """Custom exception raised for critical data quality issues."""
    pass

class DataValidator:
    REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "ticker"]

    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> bool:
        """Verifies that all required columns are present in the DataFrame."""
        missing = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")
        return True

    @classmethod
    def check_nulls(cls, df: pd.DataFrame, threshold_pct: float = 0.05) -> pd.DataFrame:
        """Checks for missing (null) values. 
        If below threshold, interpolates them. If above, raises an error.
        """
        null_counts = df.isnull().sum()
        total_rows = len(df)
        
        for col in cls.REQUIRED_COLUMNS:
            null_pct = null_counts.get(col, 0) / total_rows
            if null_pct > 0:
                if null_pct > threshold_pct:
                    raise DataValidationError(
                        f"Column '{col}' has {null_pct:.2%} missing values, exceeding threshold of {threshold_pct:.2%}"
                    )
                else:
                    logger.warning(f"Column '{col}' has {null_pct:.2%} missing values. Interpolating...")
                    if col == "timestamp" or col == "ticker":
                        raise DataValidationError(f"Cannot interpolate missing metadata column: '{col}'")
                    # Forward fill then backward fill numeric columns
                    df[col] = df[col].ffill().bfill()
        return df

    @classmethod
    def validate_values(cls, df: pd.DataFrame) -> bool:
        """Validates that OHLCV prices and volumes are logically consistent."""
        # Prices and volume must be non-negative
        for col in ["open", "high", "low", "close", "volume"]:
            if (df[col] < 0).any():
                invalid_rows = df[df[col] < 0]
                raise DataValidationError(
                    f"Negative values detected in column '{col}' at indices: {invalid_rows.index.tolist()}"
                )
        
        # High must be greater than or equal to open, close, and low
        violations_high_low = df["high"] < df["low"]
        if violations_high_low.any():
            invalid_rows = df[violations_high_low]
            raise DataValidationError(
                f"High price is less than Low price at indices: {invalid_rows.index.tolist()}"
            )
            
        violations_high_close = df["high"] < df["close"]
        if violations_high_close.any():
            logger.warning(f"High price is less than Close price at some indexes. Auto-correcting High to match Close.")
            df["high"] = np.maximum(df["high"], df["close"])

        violations_low_close = df["low"] > df["close"]
        if violations_low_close.any():
            logger.warning(f"Low price is greater than Close price at some indexes. Auto-correcting Low to match Close.")
            df["low"] = np.minimum(df["low"], df["close"])
            
        return True

    @classmethod
    def clean_and_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Performs full cleanup and validation checks on a DataFrame."""
        if df.empty:
            raise DataValidationError("Input DataFrame is empty.")
            
        # Copy to avoid side-effects
        cleaned_df = df.copy()
        
        cls.validate_schema(cleaned_df)
        cleaned_df = cls.check_nulls(cleaned_df)
        cls.validate_values(cleaned_df)
        
        # Sort chronologically to prevent temporal errors
        cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["timestamp"])
        cleaned_df = cleaned_df.sort_values(by="timestamp").reset_index(drop=True)
        
        logger.info(f"Successfully validated {len(cleaned_df)} rows of data for ticker {cleaned_df['ticker'].iloc[0]}")
        return cleaned_df
