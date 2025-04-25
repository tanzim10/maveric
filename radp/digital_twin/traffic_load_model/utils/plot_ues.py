import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from scipy.spatial import Voronoi, voronoi_plot_2d  # TODO: Check if scipy added to requirements

from radp.digital_twin.utils import constants as c

# * --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# * --- Plotting ---
def plot(tick_to_plot: int, results: Dict, site_config_data: pd.DataFrame, spatial_params: Dict):
    logger.info(f"Generating plot for tick {tick_to_plot}...")

    if Voronoi is None:
        logger.warning("Scipy missing, cannot plot Voronoi.")
        return

    if not all(k in results for k in ["trafficload_ue_data", "serving_cell_data"]):
        logger.warning("Missing data for plot.")
        return

    ue_data_tick = results["trafficload_ue_data"][results["trafficload_ue_data"]["tick"] == tick_to_plot]
    serving_cell_tick = results["serving_cell_data"][results["serving_cell_data"]["tick"] == tick_to_plot]

    if ue_data_tick.empty:
        logger.warning(f"No UE data for tick {tick_to_plot}.")
        return

    # Merge UE with their serving cell ID
    ue_data_tick = pd.merge(ue_data_tick, serving_cell_tick[["ue_id", "serving_cell_id"]], on="ue_id", how="left")

    # Column accessors
    COL_LAT = getattr(c, "LAT", "lat")
    COL_LON = getattr(c, "LON", "lon")
    COL_CELL_LAT = getattr(c, "CELL_LAT", "cell_lat")
    COL_CELL_LON = getattr(c, "CELL_LON", "cell_lon")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot Voronoi if possible
    if len(site_config_data) >= 4:
        points = site_config_data[[COL_CELL_LON, COL_CELL_LAT]].values
        try:
            vor = Voronoi(points)
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors="gray", lw=1, line_alpha=0.6, point_size=0)
        except Exception as e:
            logger.warning(f"Voronoi plot failed: {e}")

    # Plot cell towers
    ax.scatter(
        site_config_data[COL_CELL_LON],
        site_config_data[COL_CELL_LAT],
        marker="^",  # type: ignore
        c="red",
        label="Cell Towers",
        s=60,
        zorder=10,  # type: ignore
    )

    # Plot UEs per serving cell
    unique_serving = ue_data_tick["serving_cell_id"].dropna().unique()
    cmap = cm.get_cmap("tab20", max(1, len(unique_serving)))

    handles = [plt.Line2D([0], [0], marker="^", color="w", label="Cell Towers", mfc="red", ms=10)]  # type: ignore
    labels = ["Cell Towers"]

    for i, cell_id in enumerate(unique_serving):
        cell_ues = ue_data_tick[ue_data_tick["serving_cell_id"] == cell_id]
        if not cell_ues.empty:
            ax.scatter(cell_ues[COL_LON], cell_ues[COL_LAT], color=cmap(i), alpha=0.6, s=15, zorder=5)
            label = f"UEs (Cell {cell_id})"
            if label not in labels:
                handles.append(plt.Line2D([0], [0], marker="o", color=cmap(i), linestyle="", ms=6))  # type: ignore
                labels.append(label)

    # UEs without serving cell
    ues_no_serve = ue_data_tick[ue_data_tick["serving_cell_id"].isna()]
    if not ues_no_serve.empty:
        ax.scatter(ues_no_serve[COL_LON], ues_no_serve[COL_LAT], c="black", marker="x", s=15, zorder=5)  # type: ignore
        label = "UEs (No Serving Cell)"
        if label not in labels:
            handles.append(plt.Line2D([0], [0], marker="x", color="black", linestyle="", ms=6))  # type: ignore
            labels.append(label)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"UE Distribution and Serving Cells (Tick {tick_to_plot})")
    ax.legend(handles=handles, labels=labels, loc="best", fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save plot
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, f"ue_distribution_tick_{tick_to_plot}.png")
    try:
        plt.savefig(filename)
        logger.info(f"Plot saved: {filename}")
    except Exception as e:
        logger.error(f"Save plot failed {filename}: {e}")
    finally:
        plt.close(fig)
