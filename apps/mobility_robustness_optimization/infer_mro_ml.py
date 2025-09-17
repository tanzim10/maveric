import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from radp.digital_twin.utils.cell_selection import perform_attachment_hyst_ttt

from .mobility_robustness_optimization import calculate_mro_metric


def infer_mro(
    xgboost_pickle_path: str,
    full_data_w_power: pd.DataFrame,
    n_epochs: int = 20
) -> Tuple[float, int]:
    """
    Uses trained XGBoost model and preprocessed data to find optimal hyst and ttt values.
    
    Args:
        xgboost_pickle_path: Path to the trained XGBoost pickle file
        full_data_w_power: Preprocessed simulation data from preprocess_mro
        n_epochs: Number of optimization epochs
        
    Returns:
        Tuple[float, int]: Optimal hysteresis and TTT values (best_hyst, best_ttt)
    """
    # Load the trained model and parameters
    with open(xgboost_pickle_path, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    hyst_range = model_data["hyst_range"]
    ttt_range = model_data["ttt_range"]
    rlf_threshold = model_data["rlf_threshold"]
    
    # Initialize tracking variables
    X, y = [], []
    best_y = float('-inf')
    best_idx = 0
    
    # Generate some initial evaluation points
    init_samples = 3
    for _ in range(init_samples):
        hyst = np.random.uniform(hyst_range[0], hyst_range[1])
        ttt = np.random.randint(ttt_range[0], ttt_range[1])
        attached_df = perform_attachment_hyst_ttt(full_data_w_power, hyst, ttt, rlf_threshold)
        metric = calculate_mro_metric(attached_df)
        X.append([hyst, ttt])
        y.append(metric)
        
        if metric > best_y:
            best_y = metric
            best_idx = len(y) - 1
    
    X = np.array(X)
    y = np.array(y)
    
    # Optimization loop using the trained model
    for _ in range(n_epochs):
        # Generate candidate points
        cand_hyst = np.random.uniform(hyst_range[0], hyst_range[1], size=100)
        cand_ttt = np.random.randint(ttt_range[0], ttt_range[1], size=100)
        candidates = np.column_stack([cand_hyst, cand_ttt])
        
        # Use model to predict and find best candidate
        scores = model.predict(candidates)
        idx = int(np.argmax(scores))
        hyst, ttt = candidates[idx]
        ttt = int(round(ttt))
        
        # Evaluate the selected candidate
        attached_df = perform_attachment_hyst_ttt(full_data_w_power, hyst, ttt, rlf_threshold)
        metric = calculate_mro_metric(attached_df)
        
        # Update tracking
        X = np.vstack([X, [hyst, ttt]])
        y = np.append(y, metric)
        
        if metric > best_y:
            best_y = metric
            best_idx = len(y) - 1
    
    best_hyst = float(X[best_idx, 0])
    best_ttt = int(round(X[best_idx, 1]))
    
    print(f"\nOptimized Hyst: {best_hyst},\nOptimized TTT: {best_ttt}")
    return best_hyst, best_ttt