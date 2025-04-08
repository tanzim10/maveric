import json
from typing import Dict

import pandas as pd
import numpy as np

from radp.digital_twin.utils import constants as c
from radp.digital_twin.utils.gis_tools import GISTools
from radp.digital_twin.traffic_load_model.city_dt_gen import city_digitaltwin_generator


def _radp_model_rftwin(
    trafficload_ue_data: pd.DataFrame,
    _site_config_data: pd.DataFrame,
    path_loss_exponent: float = 3.5,
    ref_rx_power: float = -50,
) -> pd.DataFrame:
    """Calculates received power for each UE from each cell."""
    ue_rxpower_data = []
    for _, ue_row in trafficload_ue_data.iterrows():
        for _, cell_row in _site_config_data.iterrows():
            distance = GISTools.dist(
                (ue_row[c.LAT], ue_row[c.LON]),
                (cell_row[c.CELL_LAT], cell_row[c.CELL_LON]),
            )
            rx_power = (
                cell_row[c.CELL_TXPWR_DBM]
                + ref_rx_power
                - 10 * path_loss_exponent * np.log10(distance)
                if distance > 0.001
                else ref_rx_power
            )
            ue_rxpower_data.append(
                {
                    "tick": ue_row["tick"],
                    "ue_id": ue_row["ue_id"],
                    "cell_id": cell_row[c.CELL_ID],
                    "rx_power": rx_power,  # Keep string for consistency
                }
            )
    return pd.DataFrame(ue_rxpower_data)

def _calculate_received_power(
    self, distance_km: float, frequency_mhz: int
) -> float:
    """
    Calculate received power using the Free-Space Path Loss (FSPL) model.
    """
    # Convert distance from kilometers to meters
    distance_m = distance_km * 1000

    # Calculate Free-Space Path Loss (FSPL) in dB
    fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_mhz) - 27.55

    # Calculate and return the received power in dBm
    received_power_dbm = self.tx_power_dbm - fspl_db
    return received_power_dbm

def _radp_metric_trafficload(
    trafficload_ue_data: pd.DataFrame,
    ue_rxpower_data: pd.DataFrame,
) -> Dict[int, float]:
    """Calculates the traffic load metric (standard deviation of UE counts)."""
    ue_rxpower_data = ue_rxpower_data.loc[
        ue_rxpower_data.groupby(["tick", "ue_id"])["rx_power"].idxmax()
    ]
    ue_data_with_serving_cell = pd.merge(
        trafficload_ue_data,
        ue_rxpower_data[["tick", "ue_id", "cell_id"]],
        on=["tick", "ue_id"],
        how="left",
    )
    ue_data_with_serving_cell.rename(
        columns={"cell_id_y": "serving_cell_id"}, inplace=True
    )
    ue_counts_per_cell = (
        ue_data_with_serving_cell.groupby(["tick", "serving_cell_id"])
        .size()
        .reset_index(name="ue_count")
    )

    std_dev_per_tick = {}
    for tick in ue_counts_per_cell["tick"].unique():
        tick_data = ue_counts_per_cell[ue_counts_per_cell["tick"] == tick]
        std_dev_per_tick[tick] = tick_data["ue_count"].std()

    return std_dev_per_tick


def trafficload_metric_generator(
    trafficload_ue_data: pd.DataFrame,
    ue_rxpower_data: pd.DataFrame,
) -> Dict[int, float]:
    """Wrapper for _radp_metric_trafficload."""
    return _radp_metric_trafficload(trafficload_ue_data, ue_rxpower_data)


def energyload_metric_generator(
    trafficload_ue_data: pd.DataFrame,
    _site_config_data: pd.DataFrame,
    ue_rxpower_data: pd.DataFrame,
    rx_power_threshold: float = -90,
) -> Dict[int, float]:
    """Calculates the energy load metric (proportion of cells that can be turned off)."""
    ue_rxpower_data = ue_rxpower_data.loc[
        ue_rxpower_data.groupby(["tick", "ue_id"])["rx_power"].idxmax()
    ]
    ue_data_with_serving_cell = pd.merge(
        trafficload_ue_data,
        ue_rxpower_data[["tick", "ue_id", "cell_id"]],
        on=["tick", "ue_id"],
        how="left",
    )

    ue_data_with_serving_cell.rename(
        columns={"cell_id_y": "serving_cell_id"}, inplace=True
    )

    total_cells = len(_site_config_data)
    cells_off_per_tick = {}

    for tick in ue_data_with_serving_cell["tick"].unique():
        ue_data_tick = ue_data_with_serving_cell[ue_data_with_serving_cell["tick"] == tick]
        cells_that_can_be_turned_off = 0
        for cell_id in _site_config_data[c.CELL_ID]:
            other_cells_rx = ue_rxpower_data[
                (ue_rxpower_data[c.CELL_ID] != cell_id) & (ue_rxpower_data["tick"] == tick)
            ]
            ues_in_cell = ue_data_tick[ue_data_tick["serving_cell_id"] == cell_id]
            can_turn_off = True
            if len(ues_in_cell) > 0:
                for ue_id in ues_in_cell["ue_id"].unique():
                    ue_other_rx = other_cells_rx[other_cells_rx["ue_id"] == ue_id]
                    if (
                        ue_other_rx.empty
                        or ue_other_rx["rx_power"].max() < rx_power_threshold
                    ):
                        can_turn_off = False
                        break
            if can_turn_off:
                cells_that_can_be_turned_off += 1
        cells_off_per_tick[tick] = (
            cells_that_can_be_turned_off / total_cells #Proportion
        )

    return cells_off_per_tick

def _radp_model_trafficload(site_config_data: pd.DataFrame, total_ue: int, #Top-level function
                             spatial_params_for_city: Dict, time_params: Dict) -> Dict:
    """Top-level function to run the traffic load simulation."""
    trafficload_ue_data = city_digitaltwin_generator(total_ue, spatial_params_for_city, time_params, site_config_data)
    ue_rxpower_data = _radp_model_rftwin(trafficload_ue_data, site_config_data)
    trafficload_metric = trafficload_metric_generator(trafficload_ue_data, ue_rxpower_data)
    energyload_metric = energyload_metric_generator(trafficload_ue_data, site_config_data, ue_rxpower_data)

    return {
        "trafficload_ue_data": trafficload_ue_data,
        "ue_rxpower_data": ue_rxpower_data,
        "trafficload_metric": trafficload_metric,
        "energyload_metric": energyload_metric
    }
    
def run_tl_simulation(
    site_config_path: str,
    num_ues: int,
    spatial_params_path: str,
    time_params_path: str,
) -> Dict:
    """Wrapper function to run simulation and return metrics and data."""

    # --- Load data ---
    site_config_data = pd.read_csv(site_config_path)
    with open(spatial_params_path, "r") as f:
        spatial_params = json.load(f)
    with open(time_params_path, "r") as f:
        time_params = json.load(f)

    results = _radp_model_trafficload(site_config_data, num_ues, spatial_params, time_params)
    return results
