# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
CCO Environment Module

This module provides the environment for Coverage and Capacity Optimization (CCO).
It handles the interaction with the RADP client for running simulations and
calculating CCO metrics.
"""

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv

from apps.coverage_capacity_optimization import constants
from apps.coverage_capacity_optimization.cco_engine import CcoEngine, CcoMetric
from radp.digital_twin.utils.cell_selection import perform_attachment

load_dotenv()

from radp.client.client import RADPClient  # noqa: E402
from radp.client.helper import RADPHelper, SimulationStatus  # noqa: E402

# Configure logging
logger = logging.getLogger(__name__)


class CCOEnvironment:
    """
    Environment class for CCO optimization.

    This class manages the state of the CCO optimization problem, including:
    - Network topology
    - User equipment (UE) data
    - Configuration values
    - Interaction with the Bayesian Digital Twin via RADP client
    """

    def __init__(
        self,
        topology: pd.DataFrame,
        ue_data: pd.DataFrame,
        config: pd.DataFrame,
        bayesian_digital_twin_id: str,
        valid_configuration_values: Dict[str, List[float]],
    ):
        """
        Initialize CCO Environment.

        Args:
            topology: DataFrame with network topology (cell_lat, cell_lon, cell_el_deg, etc.)
            ue_data: DataFrame with user equipment data
            config: DataFrame with current configuration
            bayesian_digital_twin_id: ID of the trained Bayesian Digital Twin model
            valid_configuration_values: Dict of valid configuration values
                                       (e.g., {'cell_el_deg': [0, 1, 2, ..., 20]})
        """
        self.topology = topology
        self.num_cells = len(self.topology)
        self.ue_data = ue_data
        self.config = config
        self.valid_configuration_values = valid_configuration_values
        self.bayesian_digital_twin_id = bayesian_digital_twin_id

        # Initialize RADP client and helper
        self.radp_client = RADPClient()
        self.radp_helper = RADPHelper(self.radp_client)

        # Set up simulation event
        self.simulation_event: Dict[str, Any] = {
            "simulation_time_interval_seconds": 1,
            "ue_tracks": {"ue_data_id": "ue_data_1"},
            "rf_prediction": {"model_id": bayesian_digital_twin_id, "config_id": 0},
        }

        logger.info(f"CCO Environment initialized with {self.num_cells} cells")

    def calc_metric(
        self,
        lambda_: float = 0.5,
        weak_coverage_threshold: float = -90,
        over_coverage_threshold: float = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Calculate the CCO metric for the current configuration.

        Args:
            lambda_: Weight parameter for coverage metric (0 < lambda < 1)
            weak_coverage_threshold: RSRP threshold for weak coverage (dBm)
            over_coverage_threshold: SINR threshold for over coverage (dB)

        Returns:
            Tuple of (rf_dataframe, coverage_dataframe, cco_objective)
        """
        # Update the simulation event config ID
        self.simulation_event["rf_prediction"]["config_id"] += 1

        # Run simulation
        logger.debug(f"Running simulation with config_id={self.simulation_event['rf_prediction']['config_id']}")
        simulation_response = self.radp_client.simulation(
            simulation_event=self.simulation_event,
            ue_data=self.ue_data,
            config=self.config,
        )
        simulation_id = simulation_response["simulation_id"]

        # Wait for simulation to complete
        simulation_status: SimulationStatus = self.radp_helper.resolve_simulation_status(
            simulation_id,
            wait_interval=1,
            max_attempts=100,
            verbose=False,
        )

        if not simulation_status.success:
            raise Exception(f"Simulation '{simulation_id}' failed: {simulation_status.error_message}")

        # Get simulation results
        rf_dataframe = self.radp_client.consume_simulation_output(simulation_id)

        # Perform cell attachment
        cell_selected_rf_dataframe = perform_attachment(rf_dataframe, self.topology)

        # Calculate CCO coverage dataframe
        coverage_dataframe = CcoEngine.rf_to_coverage_dataframe(
            rf_dataframe=cell_selected_rf_dataframe,
            lambda_=lambda_,
            weak_coverage_threshold=weak_coverage_threshold,
            over_coverage_threshold=over_coverage_threshold,
        )

        # Calculate CCO objective value
        cco_objective = CcoEngine.get_cco_objective_value(
            coverage_dataframe=coverage_dataframe,
            active_ids_list=coverage_dataframe[constants.CELL_ID].unique(),
            cco_metric=CcoMetric.PIXEL,
        )

        logger.debug(f"CCO objective: {cco_objective:.3f}")

        return cell_selected_rf_dataframe, coverage_dataframe, cco_objective

    def update_cell_config(self, cell_id: str, param_name: str, param_value: float) -> None:
        """
        Update a configuration parameter for a specific cell.

        Args:
            cell_id: ID of the cell to update
            param_name: Name of the parameter (e.g., 'cell_el_deg')
            param_value: New value for the parameter
        """
        cell_config_index = self.config.index[self.config["cell_id"] == cell_id][0]
        self.config.loc[cell_config_index, param_name] = param_value
        logger.debug(f"Updated {param_name} for cell {cell_id} to {param_value}")

    def get_cell_config(self, cell_id: str, param_name: str) -> float:
        """
        Get the current configuration value for a specific cell.

        Args:
            cell_id: ID of the cell
            param_name: Name of the parameter (e.g., 'cell_el_deg')

        Returns:
            Current value of the parameter
        """
        cell_config_index = self.config.index[self.config["cell_id"] == cell_id][0]
        return self.config.loc[cell_config_index, param_name]

    def reset_config(self, original_config: pd.DataFrame) -> None:
        """
        Reset the configuration to original values.

        Args:
            original_config: DataFrame with original configuration
        """
        self.config = original_config.copy()
        logger.info("Configuration reset to original values")

    def get_valid_tilt_values(self) -> List[float]:
        """
        Get the list of valid tilt values for cell_el_deg.

        Returns:
            List of valid tilt values
        """
        return self.valid_configuration_values.get(constants.CELL_EL_DEG, [])

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.

        Returns:
            Dictionary with current state information
        """
        return {
            "num_cells": self.num_cells,
            "current_config": self.config.copy(),
            "topology": self.topology.copy(),
            "bayesian_digital_twin_id": self.bayesian_digital_twin_id,
        }
