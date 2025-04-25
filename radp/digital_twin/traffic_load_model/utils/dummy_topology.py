import logging
from typing import Tuple

import numpy as np
import pandas as pd

from radp.digital_twin.utils import constants as c

# * --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# * --- Topology Generation Function ---
def generate_trisectoral_topology(
    num_sites: int,
    cells_per_site: int = 3,
    lat_range: Tuple[float, float] = (40.7, 40.8),
    lon_range: Tuple[float, float] = (-74.05, -73.95),
    start_ecgi: int = 1001,
    start_enodeb_id: int = 1,
    default_tac: int = 1,
    default_freq: int = 2100,
    default_power_dbm: float = 25.0,
    azimuth_step: int = 120,
) -> pd.DataFrame:
    """Generates a DataFrame containing dummy topology data for a trisectoral cellular network."""

    logger.info(f"Generating dummy topology for {num_sites} sites with {cells_per_site} cells each.")

    topology_data = []
    current_ecgi = start_ecgi
    current_enodeb_id = start_enodeb_id

    # Define column names using constants if available, otherwise fallback strings
    COL_CELL_ID = getattr(c, "CELL_ID", "cell_id")
    COL_CELL_LAT = getattr(c, "CELL_LAT", "cell_lat")
    COL_CELL_LON = getattr(c, "CELL_LON", "cell_lon")
    COL_CELL_AZ_DEG = getattr(c, "CELL_AZ_DEG", "cell_az_deg")
    COL_CELL_TXPWR_DBM = getattr(c, "CELL_TXPWR_DBM", "cell_txpwr_dbm")
    COL_ECGI = getattr(c, "ECGI", "ecgi")
    COL_SITE_ID = getattr(c, "SITE_ID", "site_id")
    COL_CELL_NAME = getattr(c, "CELL_NAME", "cell_name")
    COL_ENODEB_ID = getattr(c, "ENODEB_ID", "enodeb_id")
    COL_TAC = getattr(c, "TAC", "tac")
    COL_CELL_CARRIER_FREQ_MHZ = getattr(c, "CELL_CARRIER_FREQ_MHZ", "cell_carrier_freq_mhz")

    for i in range(num_sites):
        site_lat = np.random.uniform(lat_range[0], lat_range[1])
        site_lon = np.random.uniform(lon_range[0], lon_range[1])
        site_id_str = f"Site{i+1}"

        for j in range(cells_per_site):
            cell_az = (j * azimuth_step) % 360
            cell_id_str = f"cell_{current_enodeb_id}_{cell_az}"
            cell_name_str = f"Cell{j+1}"

            row = {
                COL_ECGI: current_ecgi,
                COL_SITE_ID: site_id_str,
                COL_CELL_NAME: cell_name_str,
                COL_ENODEB_ID: current_enodeb_id,
                COL_CELL_AZ_DEG: cell_az,
                COL_TAC: default_tac,
                COL_CELL_LAT: site_lat,
                COL_CELL_LON: site_lon,
                COL_CELL_ID: cell_id_str,
                COL_CELL_CARRIER_FREQ_MHZ: default_freq,
                COL_CELL_TXPWR_DBM: default_power_dbm + np.random.uniform(-2, 2),
            }
            topology_data.append(row)

        current_ecgi += 1
        current_enodeb_id += 1

    df = pd.DataFrame(topology_data)
    column_order = [  # Define the exact desired order
        COL_ECGI,
        COL_SITE_ID,
        COL_CELL_NAME,
        COL_ENODEB_ID,
        COL_CELL_AZ_DEG,
        COL_TAC,
        COL_CELL_LAT,
        COL_CELL_LON,
        COL_CELL_ID,
        COL_CELL_CARRIER_FREQ_MHZ,
        COL_CELL_TXPWR_DBM,
    ]
    df = df.reindex(columns=column_order)  # Ensure order and presence of all columns

    logger.info(f"Generated dummy topology DataFrame with {len(df)} cells.")

    return df
