#!/usr/bin/env python3
"""
MRO ML Application Pipeline
Orchestrates data preprocessing, training, and inference for Mobility Robustness Optimization.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from .data_preprocess import preprocess_mro
from .train_mro_ml import train_mro
from .infer_mro_ml import infer_mro



def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("mro_ml_pipeline.log")
        ]
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration for the MRO ML pipeline."""
    default_config = {
        "model_type": "xgboost",
        "init_samples": 5,
        "n_epochs": 20,
        "log_level": "INFO"
    }

    if config_path and Path(config_path).exists():
        import json
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)

    return default_config


def run_pipeline(
    mobility_model_params: Dict[str, Dict[str, Any]],
    topology: pd.DataFrame,
    bdt_path: str,
    config: Dict[str, Any],
    new_data: Optional[pd.DataFrame] = None,
    preprocess_only: bool = False,
    train_only: bool = False,
    infer_only: bool = False,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the complete MRO ML pipeline or specific stages.

    Args:
        mobility_model_params: Parameters for mobility model
        topology: Cell topology dataframe
        bdt_path: Path to Bayesian Digital Twins pickle file
        config: Configuration dictionary
        new_data: Optional new data for boundaries
        preprocess_only: If True, only run preprocessing
        train_only: If True, only run training (requires preprocessed data)
        infer_only: If True, only run inference (requires trained model)
        model_path: Path to trained model (required for infer_only)

    Returns:
        Dict containing results from each stage
    """
    logger = logging.getLogger(__name__)
    results = {}

    # Stage 1: Data Preprocessing
    if not train_only and not infer_only:
        logger.info("Starting data preprocessing...")
        try:
            preprocessed_data = preprocess_mro(
                mobility_model_params=mobility_model_params,
                topology=topology,
                bdt_path=bdt_path,
                new_data=new_data
            )
            results["preprocessed_data"] = preprocessed_data
            logger.info(f"Preprocessing completed. Data shape: {preprocessed_data.shape}")

            # Save preprocessed data
            preprocessed_path = "preprocessed_mro_data.csv"
            preprocessed_data.to_csv(preprocessed_path, index=False)
            results["preprocessed_data_path"] = preprocessed_path
            logger.info(f"Preprocessed data saved to: {preprocessed_path}")

            if preprocess_only:
                return results

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    # Load preprocessed data if needed for training/inference
    if train_only or infer_only:
        if "preprocessed_data" not in results:
            preprocessed_path = "preprocessed_mro_data.csv"
            if not Path(preprocessed_path).exists():
                raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_path}")
            preprocessed_data = pd.read_csv(preprocessed_path)
            results["preprocessed_data"] = preprocessed_data
            logger.info(f"Loaded preprocessed data from: {preprocessed_path}")

    # Stage 2: Model Training
    if not infer_only:
        logger.info("Starting model training...")
        try:
            model_pickle_path = train_mro(
                full_data_w_power=results["preprocessed_data"],
                model_type=config["model_type"],
                init_samples=config["init_samples"]
            )
            results["model_path"] = model_pickle_path
            logger.info(f"Training completed. Model saved to: {model_pickle_path}")

            if train_only:
                return results

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    # Stage 3: Inference
    if not preprocess_only and not train_only:
        logger.info("Starting inference...")
        try:
            # Use provided model path or the one from training
            inference_model_path = model_path if model_path else results.get("model_path")
            if not inference_model_path:
                raise ValueError("No model path provided for inference")

            if not Path(inference_model_path).exists():
                raise FileNotFoundError(f"Model file not found: {inference_model_path}")

            best_hyst, best_ttt = infer_mro(
                xgboost_pickle_path=inference_model_path,
                full_data_w_power=results["preprocessed_data"],
                n_epochs=config["n_epochs"]
            )

            results["optimal_hyst"] = best_hyst
            results["optimal_ttt"] = best_ttt
            logger.info(f"Inference completed. Optimal Hyst: {best_hyst}, Optimal TTT: {best_ttt}")

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise

    return results


def main():
    """Main entry point for the MRO ML application."""
    parser = argparse.ArgumentParser(description="MRO ML Pipeline Application")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--topology", type=str, required=True, help="Path to topology CSV file")
    parser.add_argument("--bdt-path", type=str, required=True, help="Path to BDT pickle file")
    parser.add_argument("--mobility-params", type=str, help="Path to mobility params JSON file")
    parser.add_argument("--new-data", type=str, help="Path to new data CSV file")
    parser.add_argument("--preprocess-only", action="store_true", help="Only run preprocessing")
    parser.add_argument("--train-only", action="store_true", help="Only run training")
    parser.add_argument("--infer-only", action="store_true", help="Only run inference")
    parser.add_argument("--model-path", type=str, help="Path to trained model (for inference only)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = load_config(args.config)
        config["log_level"] = args.log_level

        # Load required data
        topology = pd.read_csv(args.topology)
        logger.info(f"Loaded topology with {len(topology)} cells")

        # Load mobility model parameters
        if args.mobility_params:
            import json
            with open(args.mobility_params, 'r') as f:
                mobility_model_params = json.load(f)
        else:
            # Default mobility model parameters
            mobility_model_params = {
                "ue_tracks_generation": {
                    "params": {
                        "simulation_duration_seconds": 3600,
                        "simulation_time_interval_seconds": 0.01,
                        "num_ticks": 100,
                        "num_batches": 1,
                        "lat_lon_boundaries": {
                            "min_lat": -90,
                            "max_lat": 90,
                            "min_lon": -180,
                            "max_lon": 180
                        },
                        "ue_class_distribution": {
                            "pedestrian": {"count": 10, "velocity": 1, "velocity_variance": 1},
                            "cyclist": {"count": 5, "velocity": 5, "velocity_variance": 1},
                            "car": {"count": 2, "velocity": 15, "velocity_variance": 5}
                        },
                        "gauss_markov_params": {
                            "alpha": 0.5,
                            "variance": 0.8,
                            "rng_seed": 42
                        }
                    }
                }
            }

        # Load new data if provided
        new_data = None
        if args.new_data:
            new_data = pd.read_csv(args.new_data)
            logger.info(f"Loaded new data with {len(new_data)} rows")

        # Run the pipeline
        results = run_pipeline(
            mobility_model_params=mobility_model_params,
            topology=topology,
            bdt_path=args.bdt_path,
            config=config,
            new_data=new_data,
            preprocess_only=args.preprocess_only,
            train_only=args.train_only,
            infer_only=args.infer_only,
            model_path=args.model_path
        )

        # Print results summary
        logger.info("Pipeline execution completed successfully!")
        logger.info("Results Summary:")
        for key, value in results.items():
            if key.endswith("_data"):
                continue  # Skip large dataframes
            logger.info(f"  {key}: {value}")

        return results

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()