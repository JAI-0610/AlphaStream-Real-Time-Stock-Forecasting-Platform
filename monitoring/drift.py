import numpy as np
import logging
from scipy.stats import ks_2samp
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("drift_monitor")

class DriftDetector:
    def __init__(self, p_value_threshold: float = 0.05):
        self.p_value_threshold = p_value_threshold

    def detect_drift(self, reference_data: np.ndarray, live_data: np.ndarray) -> Dict[str, Any]:
        """Performs a two-sample Kolmogorov-Smirnov test to detect distribution drift.
        
        Null Hypothesis (H0): The two samples are drawn from the same distribution.
        Alternative Hypothesis (H1): The two samples are drawn from different distributions.
        
        Args:
            reference_data: Array of historical baseline values (e.g. training set features or predictions)
            live_data: Array of recent live values (e.g. streaming inference features or predictions)
        """
        # Strip NaN/Inf
        ref = reference_data[np.isfinite(reference_data)]
        live = live_data[np.isfinite(live_data)]
        
        if len(ref) < 20 or len(live) < 20:
            # Insufficient samples to run a reliable test
            return {
                "drift_detected": False,
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "status": "INSUFFICIENT_DATA"
            }
            
        statistic, p_value = ks_2samp(ref, live)
        drift_detected = p_value < self.p_value_threshold
        
        if drift_detected:
            logger.warning(
                f"Data drift detected! KS statistic: {statistic:.4f}, p-value: {p_value:.4e} (threshold: {self.p_value_threshold})"
            )
        else:
            logger.info(
                f"Distribution comparison stable. KS statistic: {statistic:.4f}, p-value: {p_value:.4f}"
            )
            
        return {
            "drift_detected": bool(drift_detected),
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "status": "SUCCESS"
        }
