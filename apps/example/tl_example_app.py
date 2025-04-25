# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# --- MODIFIED TO OUTPUT UE DATA CSV FOR CCO (PER TICK) ---
# --- Includes dummy topology generation and consistent naming ---
# --- Generates ./gen_tl_data/ and ./plots/ directories from where this script is run ---

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# * --- Path setup (USER MUST MODIFY IF NEEDED) ---
root_dir = Path().absolute().parent  # ! FIXME: this path should be absolute: home to maveric
RADP_ROOT = os.getenv("MAVERIC_ROOT", str(root_dir))
sys.path.insert(0, RADP_ROOT)

# * --- Imports from RADP library ---
from radp.digital_twin.traffic_load_model.traffic_load_model import run_traffic_simulation_and_analysis  # noqa: E402
from radp.digital_twin.traffic_load_model.utils.dummy_topology import generate_trisectoral_topology  # noqa: E402
from radp.digital_twin.traffic_load_model.utils.plot_ues import plot  # noqa: E402
from radp.digital_twin.utils import constants as c  # noqa: E402

# * --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("Successfully imported required modules and constants from the RADP library.")


# * --- Main Execution Block ---
if __name__ == "__main__":

    # * --- Configuration ---
    GEN_TL_REL_PATH = Path("./gen_tl_data")  # Directory for generated data

    GENERATE_TOPOLOGY = True  # <<< Set to False to load your file, True to generate dummy data
    SITE_CONFIG_CSV = str(GEN_TL_REL_PATH / "topology.csv")

    NUM_SITES_TO_GENERATE = 10
    CELLS_PER_SITE_TO_GENERATE = 3
    GENERATION_LAT_RANGE = (40.7, 40.8)
    GENERATION_LON_RANGE = (-74.05, -73.95)
    DEFAULT_CELL_POWER = 25.0

    SPATIAL_PARAMS_JSON = str(GEN_TL_REL_PATH / "spatial_params.json")
    TIME_PARAMS_JSON = str(GEN_TL_REL_PATH / "time_params.json")
    NUM_UES_TO_GENERATE = 500
    OUTPUT_UE_DATA_DIR = str(GEN_TL_REL_PATH / "ue_data")  # Directory for per-tick UE data CSVs

    # * --- Setup: Load or Generate Topology ---
    site_config_data = None

    if GENERATE_TOPOLOGY:
        logger.info("Generating dummy topology data...")
        site_config_data = generate_trisectoral_topology(
            num_sites=NUM_SITES_TO_GENERATE,
            cells_per_site=CELLS_PER_SITE_TO_GENERATE,
            lat_range=GENERATION_LAT_RANGE,
            lon_range=GENERATION_LON_RANGE,
            default_power_dbm=DEFAULT_CELL_POWER,
        )
        try:
            os.makedirs(os.path.dirname(SITE_CONFIG_CSV) or ".", exist_ok=True)
            site_config_data.to_csv(SITE_CONFIG_CSV, index=False)
            logger.info(f"Saved generated dummy topology to {SITE_CONFIG_CSV}")
        except Exception as e:
            logger.error(f"Could not save generated topology: {e}")
    else:
        logger.info(f"Loading topology data from {SITE_CONFIG_CSV}...")
        try:
            site_config_data = pd.read_csv(SITE_CONFIG_CSV)

            COL_CELL_ID = getattr(c, "CELL_ID", "cell_id")
            COL_CELL_LAT = getattr(c, "CELL_LAT", "cell_lat")
            COL_CELL_LON = getattr(c, "CELL_LON", "cell_lon")
            required_cols = [COL_CELL_ID, COL_CELL_LAT, COL_CELL_LON]

            if not all(col in site_config_data.columns for col in required_cols):
                missing = [col for col in required_cols if col not in site_config_data.columns]
                raise ValueError(f"Topology missing required columns: {missing}")

            COL_CELL_TXPWR_DBM = getattr(c, "CELL_TXPWR_DBM", "cell_txpwr_dbm")
            if COL_CELL_TXPWR_DBM not in site_config_data.columns:
                logger.warning(
                    f"'{COL_CELL_TXPWR_DBM}' missing. Adding default {DEFAULT_CELL_POWER} dBm for RF analysis."
                )
                site_config_data[COL_CELL_TXPWR_DBM] = DEFAULT_CELL_POWER

            logger.info(f"Loaded {len(site_config_data)} cells from {SITE_CONFIG_CSV}")
        except FileNotFoundError:
            logger.error(f"Topology file not found: {SITE_CONFIG_CSV}.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading topology from {SITE_CONFIG_CSV}: {e}")
            sys.exit(1)

    if site_config_data is None or site_config_data.empty:
        logger.error("Site config data unavailable.")
        sys.exit(1)

    # * --- Create Dummy Spatial/Time JSONs ---
    # *** Uses actual type names now if spatial_params exists ***
    if not os.path.exists(SPATIAL_PARAMS_JSON):
        logger.warning(f"{SPATIAL_PARAMS_JSON} not found. Creating dummy data.")
        num_types_generated = 3
        default_types = [f"type_{i+1}" for i in range(num_types_generated)]
        dummy_spatial_params = {
            "types": default_types,
            "proportions": list(np.random.dirichlet(np.ones(num_types_generated))),
        }
        try:
            with open(SPATIAL_PARAMS_JSON, "w") as f:
                json.dump(dummy_spatial_params, f, indent=4)
            spatial_types_for_time = default_types  # Use generated types for time weights keys
        except Exception as e:
            logger.error(f"Could not write dummy {SPATIAL_PARAMS_JSON}: {e}")
            spatial_types_for_time = []
    else:
        # Load existing types to use for time weights keys if time file needs generating
        try:
            with open(SPATIAL_PARAMS_JSON, "r") as f:
                loaded_spatial = json.load(f)
            spatial_types_for_time = loaded_spatial.get("types", [])
        except Exception as e:
            logger.error(f"Could not read {SPATIAL_PARAMS_JSON} for type names: {e}")
            spatial_types_for_time = []

    if not os.path.exists(TIME_PARAMS_JSON):
        logger.warning(f"{TIME_PARAMS_JSON} not found. Creating dummy data.")
        num_ticks = 24
        if not spatial_types_for_time:
            logger.error("Cannot create dummy time params without spatial types defined.")
            sys.exit(1)
        dummy_time_params = {
            "total_ticks": num_ticks,
            "tick_duration": 1,
            "time_weights": {  # *** Use actual type names as keys ***
                stype: list(np.random.rand(num_ticks)) for stype in spatial_types_for_time
            },
        }
        try:
            with open(TIME_PARAMS_JSON, "w") as f:
                json.dump(dummy_time_params, f, indent=4)
        except Exception as e:
            logger.error(f"Could not write dummy {TIME_PARAMS_JSON}: {e}")

    # * --- Run Simulation ---
    try:
        results = run_traffic_simulation_and_analysis(
            site_config_data=site_config_data,
            num_ues=NUM_UES_TO_GENERATE,
            spatial_params_path=SPATIAL_PARAMS_JSON,
            time_params_path=TIME_PARAMS_JSON,
        )
    except Exception as e:
        logger.exception(f"Traffic simulation failed: {e}")
        sys.exit(1)

    if "error" in results or "trafficload_ue_data" not in results or results["trafficload_ue_data"].empty:
        logger.error("Failed to generate valid UE data from traffic simulation.")
        sys.exit(1)

    # * --- Save UE Data Per Tick ---
    logger.info(f"Saving generated UE data per tick to directory: {OUTPUT_UE_DATA_DIR}")
    os.makedirs(OUTPUT_UE_DATA_DIR, exist_ok=True)
    generated_ue_data_all_ticks = results["trafficload_ue_data"]
    saved_files_count = 0
    failed_saves = 0
    COL_LAT = getattr(c, "LAT", "lat")  # Get constants/defaults
    COL_LON = getattr(c, "LON", "lon")

    try:
        required_output_cols = ["ue_id", COL_LON, COL_LAT, "tick"]
        if not all(col in generated_ue_data_all_ticks.columns for col in required_output_cols):
            missing = [col for col in required_output_cols if col not in generated_ue_data_all_ticks.columns]
            raise ValueError(f"Generated UE data missing cols: {missing}")

        for tick in sorted(generated_ue_data_all_ticks["tick"].unique()):
            try:
                ue_data_tick_snapshot = generated_ue_data_all_ticks[generated_ue_data_all_ticks["tick"] == tick].copy()
                final_ue_data_for_cco = ue_data_tick_snapshot[["ue_id", COL_LON, COL_LAT, "tick"]].rename(
                    columns={"ue_id": "mock_ue_id", COL_LON: "lon", COL_LAT: "lat"}
                )
                final_ue_data_for_cco = final_ue_data_for_cco[["mock_ue_id", "lon", "lat", "tick"]]
                output_filename = f"generated_ue_data_for_cco_{tick}.csv"
                output_csv_path = os.path.join(OUTPUT_UE_DATA_DIR, output_filename)
                final_ue_data_for_cco.to_csv(output_csv_path, index=False)
                saved_files_count += 1
                logger.debug(f"Saved UE data for tick {tick} to: {output_csv_path}")
            except Exception as tick_e:
                logger.error(f"Failed saving UE data for tick {tick}: {tick_e}")
                failed_saves += 1

        logger.info(f"Finished saving per-tick UE data. Successful: {saved_files_count}, Failed: {failed_saves}.")

    except Exception as e:
        logger.error(f"Failed during per-tick UE data saving: {e}")

    # *  --- Optional: Print Metrics and Plot Results Per Tick ---
    logger.info("\n--- Analysis Metrics (Per Tick) ---")

    if "trafficload_metric_per_tick" in results:
        print("Traffic Load Metric (Std Dev UE Counts):")
        for tick, metric in sorted(results["trafficload_metric_per_tick"].items()):
            print(f"  Tick {tick:02d}: {metric:.2f}")

    if "energyload_metric_per_tick" in results:
        print("\nEnergy Load Metric (Prop. Cells Off):")
        for tick, metric in sorted(results["energyload_metric_per_tick"].items()):
            print(f"  Tick {tick:02d}: {metric:.3f}")

    logger.info("\n--- Generating Plots Per Tick (if matplotlib available) ---")

    try:
        # Reload spatial params for plotting context if needed
        with open(SPATIAL_PARAMS_JSON, "r") as f:
            spatial_params_plot = json.load(f)

        plotted_ticks_count = 0
        for tick in sorted(generated_ue_data_all_ticks["tick"].unique()):
            logger.debug(f"Generating plot for tick {tick}...")
            plot(tick, results, site_config_data, spatial_params_plot)  # Pass original site_config
            plotted_ticks_count += 1

        logger.info(f"Successfully generated {plotted_ticks_count} plots in ./plots/ directory.")

    except NameError:  # Catch if matplotlib failed to import
        logger.warning("Matplotlib not found or failed to import. Skipping plot generation.")
    except Exception as e:
        logger.error(f"Failed during plot generation: {e}")

    logger.info("Script finished.")
