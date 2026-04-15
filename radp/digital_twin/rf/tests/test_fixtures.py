# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared test fixtures for DTModel contract and algorithm-specific tests.

Data generators here are patterned after (but independent from)
radp/digital_twin/rf/bayesian/tests/test_bayesian_engine.py so that
the fixture module has no runtime dependency on the Bayesian test file.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from radp.digital_twin.utils.gis_tools import GISTools


def get_sample_site_config_and_ue_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return small (3-cell, 15-point) sample data for unit tests."""
    site_configs_df = pd.DataFrame(
        {
            "cell_id": ["Cell1", "Cell2", "Cell3"],
            "cell_az_deg": [0, 120, 240],
            "cell_el_deg": [0, 0, 0],
            "cell_lat": [35.690555, 35.690555, 35.690555],
            "cell_lon": [139.69194, 139.69194, 139.69194],
            "cell_carrier_freq_mhz": [2100, 2100, 2100],
        }
    )

    ue_data_df = pd.DataFrame(
        {
            "cell_id": [
                "Cell1", "Cell1", "Cell1", "Cell1", "Cell1",
                "Cell2", "Cell2", "Cell2", "Cell2", "Cell2",
                "Cell3", "Cell3", "Cell3", "Cell3", "Cell3",
            ],
            "avg_rsrp": [
                -80, -70, -75, -72, -71,
                -100, -101, -102, -99, -98,
                -77, -78, -79, -80, -79,
            ],
            "lon": [
                139.699058, 139.707889, 139.700023, 139.702645, 139.702645,
                139.707067, 139.700519, 139.701644, 139.701644, 139.701644,
                139.702793, 139.703664, 139.704312, 139.704312, 139.704690,
            ],
            "lat": [
                35.644327, 35.647810, 35.643857, 35.645913, 35.645910,
                35.647007, 35.644816, 35.645196, 35.645196, 35.645198,
                35.645571, 35.645876, 35.646208, 35.646209, 35.645790,
            ],
        }
    )
    return site_configs_df, ue_data_df


def get_x_max_and_x_min() -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return typical normalization bounds for RF Digital Twin features."""
    x_max = {
        "cell_el_deg": 50,
        "cell_lat": 90,
        "cell_lon": 180,
        "distance": 5,
        "relative_bearing": 360,
        "relative_tilt": 180,
        "relative_tilt_squared": 32400,
    }
    x_min = {
        "cell_el_deg": -10,
        "cell_lat": -90,
        "cell_lon": -180,
        "distance": 0,
        "relative_bearing": 0,
        "relative_tilt": -90,
        "relative_tilt_squared": 0,
    }
    return x_max, x_min


def augment_ue_data(
    ue_data_df: pd.DataFrame, site_configs_df: pd.DataFrame
) -> None:
    """Add engineered features (log_distance, relative_bearing) to UE data in-place."""
    for i in ue_data_df.index:
        site_config = site_configs_df[
            site_configs_df.cell_id == ue_data_df.at[i, "cell_id"]
        ]
        ue_data_df.at[i, "loc_x"] = ue_data_df.at[i, "lon"]
        ue_data_df.at[i, "loc_y"] = ue_data_df.at[i, "lat"]
        ue_data_df.at[i, "cell_lat"] = site_config.cell_lat.values[0]
        ue_data_df.at[i, "cell_lon"] = site_config.cell_lon.values[0]
        ue_data_df.at[i, "cell_az_deg"] = site_config.cell_az_deg.values[0]
        ue_data_df.at[i, "cell_el_deg"] = site_config.cell_el_deg.values[0]
        ue_data_df.at[i, "cell_carrier_freq_mhz"] = site_config.cell_carrier_freq_mhz.values[0]

        ue_data_df.at[i, "log_distance"] = GISTools.get_log_distance(
            ue_data_df.at[i, "cell_lat"],
            ue_data_df.at[i, "cell_lon"],
            ue_data_df.at[i, "lat"],
            ue_data_df.at[i, "lon"],
        )
        ue_data_df.at[i, "relative_bearing"] = GISTools.get_relative_bearing(
            ue_data_df.at[i, "cell_az_deg"],
            ue_data_df.at[i, "cell_lat"],
            ue_data_df.at[i, "cell_lon"],
            ue_data_df.at[i, "lat"],
            ue_data_df.at[i, "lon"],
        )


def split_training_and_test_data(
    ue_data_df: pd.DataFrame, test_size: float
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Split per-cell UE data into training and test sets."""
    cell_id_ue_data_map = {k: v for k, v in ue_data_df.groupby("cell_id")}

    cell_id_training_data_map: Dict[str, pd.DataFrame] = {}
    cell_id_test_data_map: Dict[str, pd.DataFrame] = {}

    for cell_id, ue_data in cell_id_ue_data_map.items():
        train, test = train_test_split(ue_data, test_size=test_size, random_state=42)
        cell_id_training_data_map[cell_id] = train
        cell_id_test_data_map[cell_id] = test

    return cell_id_training_data_map, cell_id_test_data_map


def build_data_in_lists(
    cell_id_data_map: Dict[str, pd.DataFrame],
) -> List[pd.DataFrame]:
    """Convert cell_id → DataFrame map to ordered list (for DTModel.train)."""
    return list(cell_id_data_map.values())


def generate_synthetic_rf_data(
    n_points: int = 1000,
    n_cells: int = 3,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic RF data for benchmarks and scalability tests.

    Returns ``(site_configs_df, ue_data_df)`` with ``n_cells`` cells and
    approximately ``n_points`` UE observations per cell.
    """
    rng = np.random.RandomState(seed)

    base_lat = 37.7749
    base_lon = -122.4194
    azimuths = np.linspace(0, 360, n_cells, endpoint=False)

    cell_ids = [f"SynCell{i}" for i in range(n_cells)]
    site_configs_df = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_az_deg": azimuths.tolist(),
            "cell_el_deg": [0] * n_cells,
            "cell_lat": [base_lat] * n_cells,
            "cell_lon": [base_lon] * n_cells,
            "cell_carrier_freq_mhz": [2100] * n_cells,
        }
    )

    rows = []
    for cell_id, az in zip(cell_ids, azimuths):
        for _ in range(n_points):
            lat = base_lat + rng.uniform(-0.05, 0.05)
            lon = base_lon + rng.uniform(-0.05, 0.05)
            dist_m = GISTools.get_log_distance(base_lat, base_lon, lat, lon)
            rsrp = -70 - dist_m * 10 + rng.normal(0, 5)
            rows.append({"cell_id": cell_id, "avg_rsrp": rsrp, "lat": lat, "lon": lon})

    ue_data_df = pd.DataFrame(rows)
    return site_configs_df, ue_data_df


def prepare_standard_data(
    test_size: float = 0.2,
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    pd.DataFrame,
    Dict[str, float],
    Dict[str, float],
]:
    """One-stop helper: load sample data, augment, split.

    Returns ``(train_map, test_map, site_configs_df, x_max, x_min)``.
    """
    site_configs_df, ue_data_df = get_sample_site_config_and_ue_data()
    site_configs_df.reset_index(drop=True, inplace=True)
    augment_ue_data(ue_data_df, site_configs_df)
    train_map, test_map = split_training_and_test_data(ue_data_df, test_size)
    x_max, x_min = get_x_max_and_x_min()
    return train_map, test_map, site_configs_df, x_max, x_min
