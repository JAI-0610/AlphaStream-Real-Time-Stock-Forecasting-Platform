import os
import logging
import yaml
import yfinance as yf
import pandas as pd
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("historical_downloader")

class HistoricalDownloader:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.tickers: List[str] = self.config["data"]["tickers"]
        self.sector_etfs: List[str] = self.config["data"]["sector_etfs"]
        self.start_date: str = self.config["data"]["start_date"]
        self.end_date: str = self.config["data"]["end_date"]
        if not self.end_date or self.end_date == "today":
            import datetime
            self.end_date = datetime.date.today().strftime("%Y-%m-%d")
        self.resolution: str = self.config["data"]["resolution"]
        self.cache_dir: str = self.config["data"]["cache_dir"]
        
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_ticker(self, ticker: str) -> pd.DataFrame:
        """Downloads historical OHLCV data for a single ticker."""
        logger.info(f"Downloading historical data for {ticker} from {self.start_date} to {self.end_date} (Resolution: {self.resolution})")
        
        try:
            # yfinance handles interval mappings (e.g. 1d, 1h, 5m)
            # Intraday limits check
            if self.resolution in ["5m", "1m", "15m", "1h"] and self.start_date:
                # yfinance only allows downloading last 60 days of 1h / 5m data
                logger.warning(f"Intraday resolution '{self.resolution}' requested. yfinance limits historical intraday data to the last 60 days. Adjusting download range.")
                ticker_data = yf.download(ticker, period="60d", interval=self.resolution)
            else:
                ticker_data = yf.download(ticker, start=self.start_date, end=self.end_date, interval=self.resolution)
                
            if ticker_data.empty:
                logger.error(f"No data returned for ticker: {ticker}")
                return pd.DataFrame()
            
            # Reset index to make Date/Datetime a column
            ticker_data = ticker_data.reset_index()
            
            # Clean up multi-index columns if present (yfinance sometimes returns multi-index)
            if isinstance(ticker_data.columns, pd.MultiIndex):
                ticker_data.columns = [col[0] if col[1] == '' or col[0] == col[1] else f"{col[0]}_{col[1]}" for col in ticker_data.columns]
            
            # Coerce column names to lowercase
            ticker_data.columns = [str(c).lower() for c in ticker_data.columns]
            
            # Remove ticker suffix if present (e.g. close_aapl -> close)
            suffix = f"_{ticker.lower()}"
            ticker_data.columns = [c[:-len(suffix)] if c.endswith(suffix) else c for c in ticker_data.columns]
            
            # Standardize date column name
            if "date" in ticker_data.columns:
                ticker_data = ticker_data.rename(columns={"date": "timestamp"})
            elif "datetime" in ticker_data.columns:
                ticker_data = ticker_data.rename(columns={"datetime": "timestamp"})
                
            # Add ticker identifier
            ticker_data["ticker"] = ticker
            
            return ticker_data
            
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {str(e)}")
            return pd.DataFrame()

    def get_data(self, ticker: str, force_download: bool = False) -> pd.DataFrame:
        """Gets data from local cache if available, otherwise downloads and caches it."""
        cache_path = os.path.join(self.cache_dir, f"{ticker}_{self.resolution}.csv")
        
        if os.path.exists(cache_path) and not force_download:
            try:
                logger.info(f"Loading {ticker} from cache at {cache_path}")
                df = pd.read_csv(cache_path)
                if not df.empty:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker}: {e}")
        
        try:
            df = self.download_ticker(ticker)
            if not df.empty:
                df.to_csv(cache_path, index=False)
                logger.info(f"Cached data for {ticker} at {cache_path}")
                return df
        except Exception as e:
            logger.error(f"Download failed for {ticker}: {e}")

        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
            
        return pd.DataFrame()

    def run_all(self, force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """Downloads and caches data for all tickers and ETFs."""
        all_data = {}
        for ticker in self.tickers + self.sector_etfs:
            df = self.get_data(ticker, force_download=force_download)
            if not df.empty:
                all_data[ticker] = df
        return all_data

if __name__ == "__main__":
    downloader = HistoricalDownloader()
    data = downloader.run_all(force_download=True)
    print(f"Downloaded {len(data)} assets: {list(data.keys())}")
