import json
import pandas as pd
import numpy as np
from typing import List, Tuple

def load_from_json(filepath: str, step_key: str = "step", loss_key: str = "loss") -> Tuple[np.ndarray, np.ndarray]:
    """Loads curve data from a JSON list of dictionaries."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    steps = np.array([d[step_key] for d in data])
    losses = np.array([d[loss_key] for d in data])
    
    # Ensure sorted order
    idx = np.argsort(steps)
    return steps[idx], losses[idx]

def load_from_csv(filepath: str, step_col: str = "step", loss_col: str = "loss") -> Tuple[np.ndarray, np.ndarray]:
    """Loads curve data from a CSV file."""
    df = pd.read_csv(filepath)
    return df[step_col].values, df[loss_col].values

def moving_average(data: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooths noisy SGD curves. 
    Crucial for GP performance to prevent over-fitting to local spikes.
    """
    if window < 2:
        return data
    return np.convolve(data, np.ones(window)/window, mode='same')

def detect_early_stopping(preds: np.ndarray, stds: np.ndarray, target_loss: float) -> int:
    """
    Returns the index where the predicted loss (with 95% confidence) 
    is likely to hit the target.
    """
    upper_bound = preds + 1.96 * stds
    indices = np.where(upper_bound <= target_loss)[0]
    return indices[0] if len(indices) > 0 else -1