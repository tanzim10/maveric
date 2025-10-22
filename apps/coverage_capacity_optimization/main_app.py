# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Main Application for Coverage and Capacity Optimization (CCO).

This module provides a command-line interface to run the complete CCO pipeline:
BDT training -> CCO training -> Prediction of optimized el_deg values.
"""

import argparse
import logging
import os
import sys
from typing import Dict

import pandas as pd

from apps.coverage_capacity_optimization.bdt_manager import BDTManager
from apps.coverage_capacity_optimization.cco_env import CCOEnvironment
from apps.coverage_capacity_optimization.cco_prediction import CCOPredictor
from apps.coverage_capacity_optimization.cco_trainer import CCOTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CCOPipeline:
    """Main application class for CCO pipeline orchestration."""

    def __init__(self, topology_path: str, training_data_path: str, prediction_data_path: str, config_path: str, model_id: str):
        """Initialize CCO Main Application."""
        self.topology_path = topology_path
        self.training_data_path = training_data_path
        self.prediction_data_path = prediction_data_path
        self.config_path = config_path
        self.model_id = model_id
        
        logger.info("CCO Main Application initialized")

    def run_full_pipeline(self, container_name: str = "radp_prod-training-1", epochs: int = 20) -> Dict:
        """Run the complete CCO pipeline."""
        logger.info("Starting full CCO pipeline")

        try:
            # Step 1: Train BDT model
            logger.info("=== Step 1: Training BDT Model ===")
            bdt_results = self._train_bdt(container_name)

            # Step 2: Train CCO model
            logger.info("=== Step 2: Training CCO Model ===")
            cco_results = self._train_cco(epochs)

            # Step 3: Generate predictions
            logger.info("=== Step 3: Generating Predictions ===")
            prediction_results = self._predict_el_deg()

            # Combine results
            pipeline_results = {
                "bdt_training": bdt_results,
                "cco_training": cco_results,
                "predictions": prediction_results,
            }

            logger.info("Full CCO pipeline completed successfully")
            return pipeline_results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _train_bdt(self, container_name: str) -> Dict:
        """Train Bayesian Digital Twin model."""
        logger.info("Training BDT model")

        # Initialize BDT manager
        model_path = f"models/bdt_{self.model_id}.pickle"
        bdt_manager = BDTManager(
            topology_path=self.topology_path, 
            training_data_path=self.training_data_path, 
            model_path=model_path
        )

        # Train model
        bdt_manager.train(model_id=self.model_id, container_name=container_name)

        logger.info("BDT model training completed")
        return {"status": "success", "model_id": self.model_id}

    def _train_cco(self, epochs: int) -> Dict:
        """Train CCO optimization model."""
        logger.info("Training CCO model")

        # Load data
        topology = pd.read_csv(self.topology_path)
        prediction_data = pd.read_csv(self.prediction_data_path)
        config = pd.read_csv(self.config_path)

        logger.info(f"Loaded UE data from {self.prediction_data_path}: {len(prediction_data)} records")

        # Set valid configuration values
        valid_configuration_values = {"cell_el_deg": [i * 1.0 for i in range(21)]}  # 0 to 20 degrees

        # Initialize environment
        environment = CCOEnvironment(
            topology=topology,
            ue_data=prediction_data,
            config=config,
            bayesian_digital_twin_id=self.model_id,
            valid_configuration_values=valid_configuration_values,
        )

        # Initialize trainer
        trainer = CCOTrainer(environment)

        # Train model
        training_results = trainer.train(num_epochs=epochs, epsilon=0.1, seed=42)

        # Save trained model
        model_path = f"models/cco_{self.model_id}.pickle"
        trainer.save_model(model_path)

        logger.info("CCO model training completed")
        return training_results

    def _predict_el_deg(self) -> Dict:
        """Predict optimized el_deg values."""
        logger.info("Generating el_deg predictions")

        # Load data for prediction
        topology = pd.read_csv(self.topology_path)
        prediction_data = pd.read_csv(self.prediction_data_path)
        config = pd.read_csv(self.config_path)
        valid_configuration_values = {"cell_el_deg": [i * 1.0 for i in range(21)]}

        # Initialize environment
        environment = CCOEnvironment(
            topology=topology,
            ue_data=prediction_data,
            config=config,
            bayesian_digital_twin_id=self.model_id,
            valid_configuration_values=valid_configuration_values,
        )

        # Initialize predictor
        model_path = f"models/cco_{self.model_id}.pickle"
        predictor = CCOPredictor(model_path)
        predictor.set_environment(environment)

        # Generate predictions
        predictions = predictor.predict_el_deg()

        # Export predictions
        output_path = f"results/el_deg_predictions_{self.model_id}.csv"
        os.makedirs("results", exist_ok=True)
        predictor.export_predictions(predictions, output_path)

        # Generate comparison with current config
        comparison = predictor.compare_with_current(predictions)
        comparison_path = f"results/el_deg_comparison_{self.model_id}.csv"
        comparison.to_csv(comparison_path, index=False)

        # Generate recommendations
        recommendations = predictor.get_cell_recommendations(predictions)
        recommendations_path = f"results/el_deg_recommendations_{self.model_id}.csv"
        recommendations.to_csv(recommendations_path, index=False)

        # Get summary
        summary = predictor.get_optimization_summary()

        results = {
            "predictions": predictions,
            "comparison": comparison,
            "recommendations": recommendations,
            "summary": summary,
            "output_files": {
                "predictions": output_path,
                "comparison": comparison_path,
                "recommendations": recommendations_path,
            },
        }

        logger.info(f"Predictions generated and saved to {output_path}")
        logger.info(f"Generated recommendations for {len(recommendations)} cells")

        return results


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Coverage and Capacity Optimization (CCO) Main Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main_app.py --topology data/topology.csv --training data/training.csv --prediction data/prediction.csv --config data/config.csv --model-id my_model
  
  # Run with custom epochs and container
  python main_app.py --topology data/topology.csv --training data/training.csv --prediction data/prediction.csv --config data/config.csv --model-id my_model --epochs 50 --container radp_prod-training-1
        """,
    )

    # Required arguments
    parser.add_argument("--topology", type=str, required=True, help="Path to topology CSV file")
    parser.add_argument("--training", type=str, required=True, help="Path to training data CSV file")
    parser.add_argument("--prediction", type=str, required=True, help="Path to prediction data CSV file")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration CSV file")
    parser.add_argument("--model-id", type=str, required=True, help="Model identifier for BDT training")

    # Optional arguments
    parser.add_argument("--container", type=str, default="radp_prod-training-1", help="Docker container name for BDT training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")

    return parser


def main():
    """Main entry point for the application."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize application
        app = CCOPipeline(
            topology_path=args.topology,
            training_data_path=args.training,
            prediction_data_path=args.prediction,
            config_path=args.config,
            model_id=args.model_id,
        )

        # Run pipeline
        results = app.run_full_pipeline(container_name=args.container, epochs=args.epochs)

        # Print summary
        logger.info("=== Pipeline Results Summary ===")
        logger.info(f"BDT Training: {results['bdt_training']['status']}")
        logger.info(f"CCO Training: {results['cco_training']['best_reward']:.3f} best reward")
        logger.info(f"Predictions: {len(results['predictions']['predictions'])} cells optimized")

        if results["predictions"]["recommendations"] is not None:
            logger.info(f"Recommendations: {len(results['predictions']['recommendations'])} cells need changes")

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()