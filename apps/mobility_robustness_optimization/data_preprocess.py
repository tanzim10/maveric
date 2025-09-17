
import logging
from typing import Any, Dict, Optional

import pandas as pd
import pickle
from pathlib import Path

from notebooks.radp_library import find_sim_boundary, get_ue_data
from notebooks.radp_library import (
    calc_relative_bearing,
    preprocess_ue_data,
)


def preprocess_mro(
    mobility_model_params: Dict[str, Dict[str, Any]],
    topology: pd.DataFrame,
    bdt_path: str,
    new_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Standalone function to preprocess MRO data by generating UE simulation data and making predictions.
    
    Args:
        mobility_model_params: Parameters for mobility model
        topology: Cell topology dataframe
        bdt_path: Path to Bayesian Digital Twins pickle file
        new_data: Optional new data for boundaries
        
    Returns:
        pd.DataFrame: Preprocessed simulation data ready for training
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    # Load Bayesian Digital Twins
    bdt_file = Path(bdt_path)
    if not bdt_file.exists():
        raise FileNotFoundError(f"BDT file not found: {bdt_path}")
        
    with open(bdt_file, 'rb') as f:
        bayesian_digital_twins = pickle.load(f)

    if not bayesian_digital_twins:
        raise ValueError("Bayesian Digital Twins are not trained. Train the models before preprocessing.")

    # Fix device attribute for unpickled models
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for cell_id, bdt in bayesian_digital_twins.items():
        if not hasattr(bdt, 'device'):
            bdt.device = device
            logger.info(f"Set device for {cell_id} to {device}")
    
    mobility_model_params = {
    "ue_tracks_generation": {
            "params": {
                "simulation_duration": 3600,
                "simulation_time_interval_seconds": 0.01,
                "num_ticks": 50,
                "num_batches": 1,
                "ue_class_distribution": {
                    "stationary": {
                        "count": 10,
                        "velocity": 0,
                        "velocity_variance": 1
                    },
                    "pedestrian": {
                        "count": 5,
                        "velocity": 2,
                        "velocity_variance": 1
                    },
                    "cyclist": {
                        "count": 5,
                        "velocity": 5,
                        "velocity_variance": 1
                    },
                    "car": {
                        "count": 12,
                        "velocity": 20,
                        "velocity_variance": 1
                    }
                },
                "lat_lon_boundaries": {
                    "min_lat": -90,
                    "max_lat": 90,
                    "min_lon": -180,
                    "max_lon": 180
                },
                "gauss_markov_params": {
                    "alpha": 0.5,
                    "variance": 0.8,
                    "rng_seed": 42,
                    "lon_x_dims": 100,
                    "lon_y_dims": 100,
                    "// TODO": "Account for supporting the user choosing the anchor_loc and cov_around_anchor.",
                    "// Current implementation": "the UE Tracks generator will not be using these values.",
                    "// anchor_loc": {},
                    "// cov_around_anchor": {}
            }
        }
    }
}

    # Determine simulation boundaries
    bounds = find_sim_boundary(topology, new_data)
    if "ue_tracks_generation" in mobility_model_params:
        if "params" in mobility_model_params["ue_tracks_generation"]:
            if "lat_lon_boundaries" in mobility_model_params["ue_tracks_generation"]["params"]:
                mobility_model_params["ue_tracks_generation"]["params"]["lat_lon_boundaries"].update(bounds)

    # Generate UE simulation data
    simulation_data = get_ue_data(mobility_model_params)
    simulation_data = simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})

    # Ensure topology cell_id format is consistent
    if topology["cell_id"].dtype == int:
        topology["cell_id"] = topology["cell_id"].apply(lambda x: f"cell_{int(x)}")

    # Make predictions using Bayesian Digital Twins (standalone implementation)
    logger.info("Making predictions using Bayesian Digital Twins...")
    
    # Use the same preprocessing pipeline as the original _predictions method
    prediction_data = preprocess_ue_data(simulation_data, topology)
    prediction_data = calc_relative_bearing(prediction_data)
    
    # Debug: Check what columns we have
    logger.info(f"Prediction data columns: {prediction_data.columns.tolist()}")
    
    # Add missing elevation angle if not present (set to 0 as default)
    if 'cell_el_deg' not in prediction_data.columns:
        prediction_data['cell_el_deg'] = 0.0
        logger.info("Added default cell_el_deg column")
    full_prediction_df = pd.DataFrame()
    
    # Loop over each 'tick' and 'cell_id' to make predictions (same as original _predictions method)
    for tick, tick_df in prediction_data.groupby("tick"):
        for cell_id, cell_df in tick_df.groupby("cell_id"):
            cell_id_str = f"cell_{cell_id}"  # Convert to string format used by BDT
            # Check if the Bayesian model for this cell_id exists
            if cell_id_str in bayesian_digital_twins:
                # Perform the Bayesian prediction
                pred_means_percell, _ = bayesian_digital_twins[cell_id_str].predict_distributed_gpmodel(
                    prediction_dfs=[cell_df]
                )
                
                # Add predictions to the cell dataframe
                cell_df = cell_df.copy()
                cell_df["pred_means"] = pred_means_percell[0]
                cell_df["tick"] = tick
                cell_df["cell_id"] = cell_id_str
                
                # Append the predictions to the full DataFrame
                full_prediction_df = pd.concat([full_prediction_df, cell_df], ignore_index=True)
            else:
                logger.warning(f"No model available for cell_id {cell_id_str}, skipping prediction.")
    
    if full_prediction_df.empty:
        raise ValueError("No predictions could be made. Check BDT models and data compatibility.")
    
    # Rename columns to match expected format
    full_prediction_df = full_prediction_df.rename(columns={"latitude": "loc_y", "longitude": "loc_x"})
    if full_prediction_df["cell_id"].dtype == object:
        full_prediction_df["cell_id"] = full_prediction_df["cell_id"].str.extract(r"(\d+)").astype(int)
    
    # Apply the same preprocessing as _preprocess_simulation_data (NOT perform_attachment)
    logger.info("Preprocessing simulation data...")
    
    # Drop unnecessary columns (same as _preprocess_simulation_data)
    columns_to_drop = ["rxpower_stddev_dbm", "rxpower_dbm", "cell_rxpwr_dbm"]
    # Only drop columns that exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in full_prediction_df.columns]
    
    full_prediction_df.drop(columns=columns_to_drop, inplace=True)
    
    # Rename columns to match expected format for MRO functions (same as _preprocess_simulation_data)
    full_prediction_df.rename(
        columns={
            "mock_ue_id": "ue_id",
            "log_distance": "distance_km",
            "pred_means": "cell_rxpower_dbm",  # This is what find_hyst_diff expects!
        },
        inplace=True,
    )
    
    # Convert cell_id formats if needed (same as _preprocess_simulation_data)
    if topology["cell_id"].dtype == object:
        topology["cell_id"] = topology["cell_id"].str.replace("cell_", "").astype(int)
    if full_prediction_df["cell_id"].dtype == object:
        full_prediction_df["cell_id"] = full_prediction_df["cell_id"].str.extract(r"(\d+)").astype(int)
    
    # TODO: Add SINR column if needed (calling _add_sinr_column)
    # For now, just return the data without SINR
    processed_data = full_prediction_df
    
    logger.info(f"Preprocessed data shape: {processed_data.shape}")
    logger.info(f"Preprocessed data columns: {processed_data.columns.tolist()}")
    return processed_data
