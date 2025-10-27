"""Integration tests for agentic mobility generation."""
import unittest
from unittest.mock import patch

import pandas as pd

from radp.digital_twin.agentic_mobility.integration import AgenticMobilityIntegration
from radp.digital_twin.mobility.ue_tracks_params import UETracksGenerationParams


class TestAgenticMobilityIntegration(unittest.TestCase):
    """Test end-to-end integration with existing RADP mobility code.

    How to Run:
    -----------
    python3 -m unittest radp/digital_twin/agentic_mobility/tests/test_integration.py
    """

    @patch("radp.digital_twin.agentic_mobility.integration.generate_mobility_params")
    def test_params_compatibility_with_ue_tracks_params(self, mock_generate):
        """Test that generated params are compatible with UETracksGenerationParams."""
        # Mock the response from generate_mobility_params
        mock_generate.return_value = {
            "status": "success",
            "radp_params": {
                "ue_tracks_generation": {
                    "params": {
                        "simulation_duration": 3600,
                        "simulation_time_interval_seconds": 0.01,
                        "num_ticks": 50,
                        "num_batches": 1,
                        "ue_class_distribution": {
                            "stationary": {"count": 5, "velocity": 0.0, "velocity_variance": 0.0},
                            "pedestrian": {"count": 25, "velocity": 1.3, "velocity_variance": 0.4},
                            "cyclist": {"count": 15, "velocity": 5.0, "velocity_variance": 0.8},
                            "car": {"count": 55, "velocity": 10.0, "velocity_variance": 3.0},
                        },
                        "lat_lon_boundaries": {
                            "min_lat": 35.6,
                            "max_lat": 35.7,
                            "min_lon": 139.7,
                            "max_lon": 139.8,
                        },
                        "gauss_markov_params": {
                            "alpha": 0.42,
                            "variance": 0.68,
                            "rng_seed": 42,
                            "lon_x_dims": 100,
                            "lon_y_dims": 100,
                        },
                    }
                }
            },
            "metadata": {
                "retry_count": 0,
                "query_intent": {"num_ues": 100, "num_ticks": 50},
            },
        }

        # Test that params can be used to create UETracksGenerationParams
        params, _ = AgenticMobilityIntegration.generate_params_only("Generate 100 UEs in Tokyo")

        # This should not raise any exceptions
        ue_tracks_params = UETracksGenerationParams(params)

        # Verify extracted attributes
        self.assertEqual(ue_tracks_params.num_UEs, 100)
        self.assertEqual(ue_tracks_params.num_ticks, 50)
        self.assertEqual(ue_tracks_params.alpha, 0.42)
        self.assertEqual(ue_tracks_params.variance, 0.68)
        self.assertEqual(ue_tracks_params.rng_seed, 42)
        self.assertEqual(ue_tracks_params.lon_x_dims, 100)
        self.assertEqual(ue_tracks_params.lon_y_dims, 100)

    @patch("radp.digital_twin.agentic_mobility.integration.UETracksGenerator")
    @patch("radp.digital_twin.agentic_mobility.integration.generate_mobility_params")
    def test_end_to_end_dataframe_format(self, mock_generate, mock_generator):
        """Test that end-to-end workflow produces correct DataFrame format."""
        # Mock generate_mobility_params
        mock_generate.return_value = {
            "status": "success",
            "radp_params": {
                "ue_tracks_generation": {
                    "params": {
                        "simulation_duration": 3600,
                        "simulation_time_interval_seconds": 0.01,
                        "num_ticks": 2,
                        "num_batches": 1,
                        "ue_class_distribution": {
                            "stationary": {"count": 1, "velocity": 0.0, "velocity_variance": 0.0},
                            "pedestrian": {"count": 1, "velocity": 1.3, "velocity_variance": 0.4},
                            "cyclist": {"count": 0, "velocity": 5.0, "velocity_variance": 0.8},
                            "car": {"count": 0, "velocity": 10.0, "velocity_variance": 3.0},
                        },
                        "lat_lon_boundaries": {
                            "min_lat": 35.6,
                            "max_lat": 35.7,
                            "min_lon": 139.7,
                            "max_lon": 139.8,
                        },
                        "gauss_markov_params": {
                            "alpha": 0.5,
                            "variance": 0.8,
                            "rng_seed": 42,
                            "lon_x_dims": 100,
                            "lon_y_dims": 100,
                        },
                    }
                }
            },
            "metadata": {
                "retry_count": 0,
                "query_intent": {"num_ues": 2, "num_ticks": 2},
            },
        }

        # Mock UETracksGenerator to return a simple DataFrame
        mock_df = pd.DataFrame(
            {
                "ue_id": [0, 0, 1, 1],
                "longitude": [139.75, 139.76, 139.77, 139.78],
                "latitude": [35.65, 35.66, 35.67, 35.68],
                "tick": [0, 1, 0, 1],
            }
        )
        mock_generator.generate_as_lon_lat_points.return_value = [mock_df]

        # Run end-to-end
        df, metadata = AgenticMobilityIntegration.generate_from_natural_language("Generate 2 UEs in Tokyo")

        # Verify DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("ue_id", df.columns)
        self.assertIn("longitude", df.columns)
        self.assertIn("latitude", df.columns)
        self.assertIn("tick", df.columns)

        # Verify metadata
        self.assertEqual(metadata["retry_count"], 0)
        self.assertEqual(metadata["query_intent"]["num_ues"], 2)

    @patch("radp.digital_twin.agentic_mobility.integration.generate_mobility_params")
    def test_error_handling_for_failed_generation(self, mock_generate):
        """Test that errors are properly handled when generation fails."""
        # Mock a failed generation
        mock_generate.return_value = {
            "status": "failed",
            "error": "Validation failed after max retries",
        }

        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            AgenticMobilityIntegration.generate_from_natural_language("Invalid query that will fail")

        self.assertIn("Parameter generation failed", str(context.exception))

    @patch("radp.digital_twin.agentic_mobility.integration.generate_mobility_params")
    def test_params_only_mode(self, mock_generate):
        """Test generate_params_only method."""
        mock_generate.return_value = {
            "status": "success",
            "radp_params": {"ue_tracks_generation": {"params": {"num_ticks": 50}}},
            "metadata": {"retry_count": 0},
        }

        params, metadata = AgenticMobilityIntegration.generate_params_only("Generate 100 UEs")

        # Verify we got params and metadata
        self.assertIn("ue_tracks_generation", params)
        self.assertIn("retry_count", metadata)
        self.assertEqual(metadata["retry_count"], 0)


if __name__ == "__main__":
    unittest.main()
