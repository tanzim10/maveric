import unittest
from unittest.mock import MagicMock

import pandas as pd

# import numpy as np
from apps.mobility_robustness_optimization.mobility_robustness_optimization import (
    BayesianDigitalTwin,
    _count_handovers,
    calculate_mro_metric,
    reattach_columns,
)
from apps.mobility_robustness_optimization.simple_mro import SimpleMRO
from notebooks.radp_library import preprocess_ue_data


class TestMobilityRobustnessOptimization(unittest.TestCase):
    def setUp(self):
        data = {
            "ue_id": [1, 1, 1, 1, 2, 2, 3, 3, 3],
            "cell_id": ["A", "RLF", "RLF", "B", "A", "A", "X", "Y", "Z"],
            "tick": [1, 2, 3, 4, 1, 2, 1, 2, 3],
        }
        self.df = pd.DataFrame(data)
        self.dummy_topology = pd.DataFrame(
            {
                "cell_id": ["cell_001", "cell_002"],
                "cell_lat": [45.0, 46.0],
                "cell_lon": [-73.0, -74.0],
                "cell_carrier_freq_mhz": [2100, 2000],
                "cell_az_deg": [120, 240],
            }
        )

        # Mocking additional data attributes used in the method
        self.prediction_data = pd.DataFrame(
            {
                "ue_id": [0, 1],
                "tick": [0, 1],
                "loc_x": [10.0, 20.0],
                "loc_y": [5.0, 6.0],
            }
        )
        self.simulation_data = pd.DataFrame(
            {
                "ue_id": [0, 1],
                "tick": [0, 1],
                "lon": [15.0, 25.0],
                "lat": [10.0, 11.0],
                "cell_id": [1, 2],
                "cell_lat": [45.0, 46.0],
                "cell_lon": [-73.0, -74.0],
                "cell_carrier_freq_mhz": [
                    1800,
                    2100,
                ],
            }
        )
        self.update_data = pd.DataFrame(
            {
                "ue_id": [0, 1],
                "tick": [0, 1],
                "loc_x": [30.0, 40.0],
                "loc_y": [12.0, 13.0],
            }
        )
        self.training_data = pd.DataFrame({"ue_id": [0, 1], "tick": [0, 1], "loc_x": [5.0, 10.0], "loc_y": [0.0, 1.0]})

        self.mobility_model_params = {
            "param1": {"value": 10, "type": "int"},
            "param2": {"value": 20, "type": "float"},
        }

        # Mock Bayesian Digital Twin
        self.mock_bdt = MagicMock(spec=BayesianDigitalTwin)

        # Instantiate MRO object
        self.mro = SimpleMRO(
            self.mobility_model_params,
            self.dummy_topology,
            bdt={"cell_001": self.mock_bdt},
        )

    def test_train_or_update_rf_twin(self):  # TODO: Implement AFTER PR
        pass

    def test_solve(self):  # TODO: Implement AFTER PR
        pass

    def test_training(self):
        mro = SimpleMRO(mobility_model_params={}, topology=self.dummy_topology)
        train_data = self.training_data.copy()
        train_data.rename(columns={"loc_x": "latitude", "loc_y": "longitude"}, inplace=True)

        # for different n_inter
        train_data = preprocess_ue_data(train_data, self.dummy_topology)

        if self.dummy_topology["cell_id"].dtype == int:
            self.dummy_topology["cell_id"] = self.dummy_topology["cell_id"].apply(lambda x: f"cell_{x}")
        if train_data["cell_id"].dtype == int:
            train_data["cell_id"] = train_data["cell_id"].apply(lambda x: f"cell_{x}")

        # Prepare the new data for training or updating
        prepared_data = mro._prepare_train_or_update_data(train_data)

        for n_iter in [5, 10, 20]:
            loss_vs_iter = mro._training(maxiter=n_iter, train_data=prepared_data)
            self.assertEqual(len(loss_vs_iter), len(self.dummy_topology["cell_id"]))
            self.assertEqual(loss_vs_iter[0].shape[0], n_iter)  # type: ignore

    def test_predictions(self):
        topology = self.dummy_topology.copy()
        topology["cell_id"] = ["cell_1", "cell_2"]
        mro = SimpleMRO(mobility_model_params=self.mobility_model_params, topology=topology)
        train_data = self.training_data.copy()
        train_data.rename(columns={"loc_x": "latitude", "loc_y": "longitude"}, inplace=True)

        train_data = preprocess_ue_data(train_data, self.dummy_topology)

        if self.dummy_topology["cell_id"].dtype == int:
            self.dummy_topology["cell_id"] = self.dummy_topology["cell_id"].apply(lambda x: f"cell_{x}")
        if train_data["cell_id"].dtype == int:
            train_data["cell_id"] = train_data["cell_id"].apply(lambda x: f"cell_{x}")

        # Prepare the new data for training or updating
        prepared_data = mro._prepare_train_or_update_data(train_data)

        # Train the models
        mro._training(20, prepared_data)

        # Perform predictions
        prediction_data = self.prediction_data.copy()
        prediction_data.rename(columns={"loc_x": "latitude", "loc_y": "longitude"}, inplace=True)
        predicted, full_prediction_df = mro._predictions(prediction_data)

        # Assertions
        self.assertEqual(predicted.shape, (2, 5))  # Check the shape of the predicted DataFrame
        self.assertEqual(full_prediction_df.shape, (4, 15))  # Check the shape of the full prediction DataFrame
        self.assertIn("pred_means", full_prediction_df.columns)  # Ensure predictions column exists
        self.assertTrue(all(full_prediction_df["cell_id"].isin(["cell_1", "cell_2"])))  # Check cell IDs

    def test_count_handovers(self):
        result = _count_handovers(self.df)
        self.assertEqual(result, 3)

    def test_reattach_columns(self):
        self.predicted_df = pd.DataFrame({"tick": [1, 2, 3, 4], "loc_x": [10, 20, 30, 40], "loc_y": [5, 6, 7, 8]})
        self.full_prediction_df = pd.DataFrame(
            {
                "mock_ue_id": [100, 101, 102, 103],
                "tick": [1, 2, 3, 4],
                "loc_x": [10, 20, 30, 40],
                "loc_y": [5, 6, 7, 8],
            }
        )
        full_prediction_df_no_match = pd.DataFrame(
            {
                "mock_ue_id": [200, 201, 202, 203],
                "tick": [1, 2, 3, 4],
                "loc_x": [50, 60, 70, 80],  # No matching loc_x values
                "loc_y": [9, 10, 11, 12],  # No matching loc_y values
            }
        )

        result = reattach_columns(self.predicted_df, self.full_prediction_df)

        self.assertTrue("ue_id" in result.columns)  # Ensure 'ue_id' column exists
        self.assertFalse("mock_ue_id" in result.columns)  # Ensure 'mock_ue_id' column is removed
        self.assertEqual(result.shape[0], self.predicted_df.shape[0])  # Ensure size matches predicted_df
        self.assertEqual(result.loc[0, "ue_id"], 100)
        self.assertEqual(result.loc[1, "ue_id"], 101)

        result = reattach_columns(self.predicted_df, full_prediction_df_no_match)
        self.assertTrue(result["ue_id"].isna().all())

    def test_calculate_mro_metric(self):
        result = calculate_mro_metric(self.df)
        self.assertEqual(result, 1.85)
