import json
import logging
from typing import Dict

import numpy as np
import pandas as pd

from radp.digital_twin.traffic_load_model.city_dt_gen import city_digitaltwin_generator
from radp.digital_twin.utils import constants as c
from radp.digital_twin.utils.gis_tools import GISTools

# * --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _calculate_received_power(distance_km: float, frequency_mhz: int, tx_power_dbm: float = 23) -> float:
    """
    Calculate received power using the Free-Space Path Loss (FSPL) model.
    """
    # Convert distance from kilometers to meters
    distance_m = distance_km * 1000

    # Calculate Free-Space Path Loss (FSPL) in dB
    fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_mhz) - 27.55

    # Calculate and return the received power in dBm
    received_power_dbm = tx_power_dbm - fspl_db
    return received_power_dbm


def _radp_model_rftwin(
    trafficload_ue_data: pd.DataFrame,
    site_config_data: pd.DataFrame,
    ref_rx_power: float = 25,  # ? should it be ref_rx or tx power? tx power not always in topology
) -> pd.DataFrame:
    """Calculates received power (dBm) from each site to each UE using free space path loss model."""

    if trafficload_ue_data.empty or site_config_data.empty:
        return pd.DataFrame()

    COL_LAT = getattr(c, "LAT", "lat")
    COL_LON = getattr(c, "LON", "lon")
    COL_CELL_LAT = getattr(c, "CELL_LAT", "cell_lat")
    COL_CELL_LON = getattr(c, "CELL_LON", "cell_lon")
    COL_CELL_ID = getattr(c, "CELL_ID", "cell_id")
    COL_CELL_TXPWR_DBM = getattr(c, "CELL_TXPWR_DBM", "cell_txpwr_dbm")

    req_ue_cols = [COL_LAT, COL_LON, "tick", "ue_id"]
    req_site_cols = [COL_CELL_LAT, COL_CELL_LON, COL_CELL_ID, COL_CELL_TXPWR_DBM]

    if not all(col in trafficload_ue_data.columns for col in req_ue_cols):
        missing = [col for col in req_ue_cols if col not in trafficload_ue_data.columns]
        logger.error(f"UE data missing cols: {missing}")
        return pd.DataFrame()

    if not all(col in site_config_data.columns for col in req_site_cols):
        missing = [col for col in req_site_cols if col not in site_config_data.columns]
        logger.error(f"Site config missing cols: {missing}")
        return pd.DataFrame()

    ue_rxpower_data = []

    for _, ue_row in trafficload_ue_data.iterrows():
        for _, cell_row in site_config_data.iterrows():
            try:
                dist_km = GISTools.dist(
                    (ue_row[COL_LAT], ue_row[COL_LON]), (cell_row[COL_CELL_LAT], cell_row[COL_CELL_LON])
                )

                rx_power = (
                    _calculate_received_power(dist_km, cell_row[c.CELL_CARRIER_FREQ_MHZ])
                    if dist_km > 0.001
                    else ref_rx_power
                )

                ue_rxpower_data.append(
                    {
                        "tick": ue_row["tick"],
                        "ue_id": ue_row["ue_id"],
                        COL_CELL_ID: cell_row[COL_CELL_ID],
                        "rx_power_dbm": rx_power,
                    }
                )

            except Exception as e:
                logger.warning(f"RxPower calc error UE {ue_row['ue_id']}, Cell {cell_row[COL_CELL_ID]}: {e}")

    return pd.DataFrame(ue_rxpower_data)


def _determine_serving_cell(ue_rxpower_data: pd.DataFrame) -> pd.DataFrame:
    """Returns strongest (serving) cell per UE based on rx_power_dbm."""
    COL_CELL_ID = getattr(c, "CELL_ID", "cell_id")

    if ue_rxpower_data.empty or "rx_power_dbm" not in ue_rxpower_data.columns:
        return pd.DataFrame(columns=["tick", "ue_id", "serving_cell_id"])

    logger.debug("Determining serving cells...")

    try:
        idx = ue_rxpower_data.groupby(["tick", "ue_id"])["rx_power_dbm"].idxmax()
        serving_cell_data = (
            ue_rxpower_data.loc[idx, ["tick", "ue_id", COL_CELL_ID]]
            .copy()
            .rename(columns={COL_CELL_ID: "serving_cell_id"})
        )
        logger.debug(f"Determined {len(serving_cell_data)} serving cell assignments.")
        return serving_cell_data
    except Exception as e:
        logger.error(f"Serving cell determination error: {e}")
        return pd.DataFrame(columns=["tick", "ue_id", "serving_cell_id"])


def radp_metric_trafficload(trafficload_ue_data: pd.DataFrame, serving_cell_data: pd.DataFrame) -> Dict[int, float]:
    """
    Computes the standard deviation of UE load across cells for each tick,
    indicating traffic load imbalance.
    """
    if trafficload_ue_data.empty or serving_cell_data.empty:
        return {}

    ue_data_merged = pd.merge(
        trafficload_ue_data[["tick", "ue_id"]], serving_cell_data, on=["tick", "ue_id"], how="left"
    ).dropna(subset=["serving_cell_id"])

    if ue_data_merged.empty:
        return {}

    ue_counts = ue_data_merged.groupby(["tick", "serving_cell_id"]).size()
    std_per_tick = ue_counts.groupby(level="tick").std().fillna(0)

    return std_per_tick.to_dict()


def energyload_metric_generator(
    serving_cell_data: pd.DataFrame,
    ue_rxpower_data: pd.DataFrame,
    site_config_data: pd.DataFrame,
    rx_power_threshold: float = -90,
) -> Dict[int, float]:
    """
    Calculates the fraction of base stations that could be turned off per tick
    without degrading UE service quality below the threshold.
    """
    COL_CELL_ID = getattr(c, "CELL_ID", "cell_id")

    if serving_cell_data.empty or ue_rxpower_data.empty or site_config_data.empty:
        return {}

    all_cell_ids = site_config_data[COL_CELL_ID].unique()
    total_cells = len(all_cell_ids)
    if total_cells == 0:
        return {}

    cells_off_per_tick = {}

    for tick in serving_cell_data["tick"].unique():
        serving_tick = serving_cell_data[serving_cell_data["tick"] == tick]
        rx_tick = ue_rxpower_data[ue_rxpower_data["tick"] == tick]

        can_be_off_count = 0

        for cell_id in all_cell_ids:
            ues_served = serving_tick[serving_tick["serving_cell_id"] == cell_id]["ue_id"]

            # Cell is idle — can definitely be off
            if ues_served.empty:
                can_be_off_count += 1
                continue

            # Filter rx data: other cells, only these UEs
            rx_others = rx_tick[(rx_tick[COL_CELL_ID] != cell_id) & (rx_tick["ue_id"].isin(ues_served))]

            max_power_others = rx_others.groupby("ue_id")["rx_power_dbm"].max()

            # Check if all served UEs are still covered by other cells
            try:
                if max_power_others.loc[ues_served].ge(rx_power_threshold).all():
                    can_be_off_count += 1
            except KeyError:
                # Not all UEs had alt cell data, be conservative
                continue

        cells_off_per_tick[tick] = can_be_off_count / total_cells

    return cells_off_per_tick


# * --- Main traffic simulation ---
def run_traffic_simulation_and_analysis(
    site_config_data: pd.DataFrame,
    num_ues: int,
    spatial_params_path: str,
    time_params_path: str,
) -> Dict:
    logger.info("--- Starting Traffic Simulation and Analysis ---")

    # Load configs
    try:
        with open(spatial_params_path, "r") as f:
            spatial_params = json.load(f)
        with open(time_params_path, "r") as f:
            time_params = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON config: {e}")
        raise

    # Validate time_weights vs spatial types
    expected_keys = set(spatial_params.get("types", []))
    actual_keys = set(time_params.get("time_weights", {}).keys())
    if expected_keys != actual_keys:
        logger.warning(f"Mismatch between spatial types {expected_keys} and time_weights keys {actual_keys}!")
        # Optional: raise error to enforce strict config correctness
        # raise ValueError("Mismatch between spatial types and time_weights keys.")

    # Generate UE data
    ue_data = city_digitaltwin_generator(num_ues, spatial_params, time_params, site_config_data)
    if ue_data.empty:
        return {"error": "UE generation failed"}

    # Ensure power column is available for analysis
    site_cfg_analysis = site_config_data.copy()
    COL_CELL_TXPWR_DBM = getattr(c, "CELL_TXPWR_DBM", "cell_txpwr_dbm")
    if COL_CELL_TXPWR_DBM not in site_cfg_analysis.columns:
        logger.warning(f"'{COL_CELL_TXPWR_DBM}' missing in site config. Using default value of 25 dBm.")
        site_cfg_analysis[COL_CELL_TXPWR_DBM] = 25.0  # Default tx power

    # RF Twin: Rx Power and Serving Cell Mapping
    rx_power = _radp_model_rftwin(ue_data, site_cfg_analysis)
    serving_cells = _determine_serving_cell(rx_power)

    # Metrics
    traffic_metric = radp_metric_trafficload(ue_data, serving_cells)
    energy_metric = energyload_metric_generator(serving_cells, rx_power, site_cfg_analysis)

    logger.info("--- Traffic Simulation and Analysis Finished ---")

    return {
        "trafficload_ue_data": ue_data,
        "ue_rxpower_data": rx_power,
        "serving_cell_data": serving_cells,
        "trafficload_metric_per_tick": traffic_metric,
        "energyload_metric_per_tick": energy_metric,
    }
