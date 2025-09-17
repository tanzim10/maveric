import pickle
from typing import Any

import numpy as np
import pandas as pd
from radp.digital_twin.utils.cell_selection import find_hyst_diff, perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD

from .mobility_robustness_optimization import calculate_mro_metric


def train_mro(
    full_data_w_power: pd.DataFrame,
    model_type: str = "xgboost",
    init_samples: int = 5,
    **kwargs
) -> str:
    """
    Trains an XGBoost model for MRO optimization using the preprocessed data.
    
    Args:
        full_data_w_power: Preprocessed simulation data from preprocess_mro
        model_type: Type of model to use (default: "xgboost")
        init_samples: Number of initial samples for training
        **kwargs: Additional parameters
        
    Returns:
        str: Path to the saved pickle file containing the trained model
    """
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError("xgboost is required for model_type='xgboost'") from e
        model = XGBRegressor(objective="reg:squarederror")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Calculate ranges for hyperparameters
    rlf_threshold = RLF_THRESHOLD
    max_diff = find_hyst_diff(full_data_w_power)
    num_ticks = full_data_w_power["tick"].nunique()
    hyst_range = [0, max_diff]
    ttt_range = [2, max(3, num_ticks + 1)]

    # Generate training data
    X, y = [], []
    for _ in range(init_samples):
        hyst = np.random.uniform(hyst_range[0], hyst_range[1])
        ttt = np.random.randint(ttt_range[0], ttt_range[1])
        attached_df = perform_attachment_hyst_ttt(full_data_w_power, hyst, ttt, rlf_threshold)
        metric = calculate_mro_metric(attached_df)
        X.append([hyst, ttt])
        y.append(metric)

    X = np.array(X)
    y = np.array(y)
    
    # Train the model
    model.fit(X, y)
    
    # Save the model to pickle file
    pickle_path = "trained_xgboost_model.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump({
            "model": model,
            "hyst_range": hyst_range,
            "ttt_range": ttt_range,
            "rlf_threshold": rlf_threshold
        }, f)
    
    return pickle_path