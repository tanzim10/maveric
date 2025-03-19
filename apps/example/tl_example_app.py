import sys
import json
from typing import Dict, List

import pandas as pd
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

RADP_ROOT = ""
sys.path.insert(0, RADP_ROOT)

from radp.digital_twin.mobility.mobility import RandomWaypoint
from radp.digital_twin.utils import constants as c
from radp.digital_twin.utils.gis_tools import GISTools





# --- Helper Functions ---

def space_boundary(cell_topology_data: pd.DataFrame) -> Dict:
    """Calculates the bounding box of the simulation area, with a buffer."""
    min_lat = cell_topology_data[c.CELL_LAT].min()
    max_lat = cell_topology_data[c.CELL_LAT].max()
    min_lon = cell_topology_data[c.CELL_LON].min()
    max_lon = cell_topology_data[c.CELL_LON].max()
    lat_buffer = (max_lat - min_lat) * 0.1
    lon_buffer = (max_lon - min_lon) * 0.1
    return {
        "spacebound_long_1": min_lon - lon_buffer,
        "spacebound_lat_1": min_lat - lat_buffer,
        "spacebound_long_2": max_lon + lon_buffer,
        "spacebound_lat_2": min_lat - lat_buffer,
        "spacebound_long_3": max_lon + lon_buffer,
        "spacebound_lat_3": max_lat + lat_buffer,
        "spacebound_long_4": min_lon - lon_buffer,
        "spacebound_lat_4": max_lat + lat_buffer,
    }

def _space_component_list_generator(space_boundary: Dict, spatial_params: Dict) -> Dict:
    """Generates a list of space components with their relative proportions."""
    space_types = spatial_params["types"]
    proportions = spatial_params["proportions"]

    if len(space_types) != len(proportions):
        raise ValueError("The number of space types and proportions must match.")
    if not np.isclose(sum(proportions), 1.0):
        raise ValueError("Proportions must sum to 1.0")

    space_list = {f"_space{i+1}": prop for i, prop in enumerate(proportions)} #Create a dictionary
    return space_list


def _create_spatial_cells(cell_topology_data: pd.DataFrame, space_list: Dict, space_bound: Dict) -> List[Dict]:
    """Creates a list of spatial cells based on Voronoi regions and space types.
       This is now a separate, reusable function.
    """
    points = cell_topology_data[[c.CELL_LON, c.CELL_LAT]].values
    vor = Voronoi(points)

    spatial_cells = []
    for region_idx in vor.regions:
        if not region_idx or -1 in region_idx: #Skip invalid regions
            continue

        region_vertices = vor.vertices[region_idx] #Get vertices

        # --- Bounding Box Check (Important for edge cases!) ---
        min_lon, min_lat = region_vertices.min(axis=0)
        max_lon, max_lat = region_vertices.max(axis=0)
        if (
            min_lon < space_bound["spacebound_long_1"]
            or max_lon > space_bound["spacebound_long_3"]
            or min_lat < space_bound["spacebound_lat_1"]
            or max_lat > space_bound["spacebound_lat_3"]
        ):
            continue #Skip regions outside the defined space

        space_type = np.random.choice(
            list(space_list.keys()),
            p=[x / sum(space_list.values()) for x in space_list.values()],
        )

        spatial_cells.append(
            { #Convert to list
                "bounds": region_vertices.tolist(),
                "type": space_type, #Unique ID for each cell
                "cell_id": len(spatial_cells),
            }
        )
    return spatial_cells


def _city_digitaltwin_generator(
    total_ue: int,
    spatial_params_for_city: Dict,
    time_params: Dict,
    cell_topology_data: pd.DataFrame,
) -> pd.DataFrame:
    """Generates UE data over time, with improved UE distribution."""

    space_bound = space_boundary(cell_topology_data)
    space_list = _space_component_list_generator(space_bound, spatial_params_for_city)
    spatial_cells = _create_spatial_cells(cell_topology_data, space_list, space_bound)
    total_ticks = time_params["total_ticks"]
    tick_duration = time_params["tick_duration"]
    ue_data = []

    for tick in range(total_ticks):
        for ue_id in range(total_ue):
            weights = [
                time_params.get("time_weights", {}).get(cell["type"], [1.0] * total_ticks)[
                    tick
                ]
                for cell in spatial_cells
            ]
            probs = np.array(weights) / np.sum(weights)
            chosen_cell = np.random.choice(spatial_cells, p=probs)
            polygon = Polygon(chosen_cell["bounds"])

            min_lon, min_lat, max_lon, max_lat = polygon.bounds
            while True:
                ue_lon = np.random.uniform(min_lon, max_lon)
                ue_lat = np.random.uniform(min_lat, max_lat)
                point = Point(ue_lon, ue_lat)
                if polygon.contains(point):
                    break  # Valid point found

            # --- Store the UE data ---
            ue_data.append({ #Store the UE data
                "tick": tick,
                "ue_id": ue_id,
                "lat": ue_lat,
                "lon": ue_lon,
                "space_type": chosen_cell["type"],
                "cell_id": chosen_cell["cell_id"],
            })

    return pd.DataFrame(ue_data)




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


def _radp_metric_trafficload(
    trafficload_ue_data: pd.DataFrame,
    _site_config_data: pd.DataFrame,
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
    _site_config_data: pd.DataFrame,
    ue_rxpower_data: pd.DataFrame,
) -> Dict[int, float]:
    """Wrapper for _radp_metric_trafficload."""
    return _radp_metric_trafficload(trafficload_ue_data, _site_config_data, ue_rxpower_data)


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

def _radp_model_trafficload(_site_config_data: pd.DataFrame, total_ue: int, #Top-level function
                             spatial_params_for_city: Dict, time_params: Dict) -> Dict:
    """Top-level function to run the traffic load simulation."""
    trafficload_ue_data = _city_digitaltwin_generator(total_ue, spatial_params_for_city, time_params, _site_config_data)
    ue_rxpower_data = _radp_model_rftwin(trafficload_ue_data, _site_config_data)
    trafficload_metric = trafficload_metric_generator(trafficload_ue_data, _site_config_data, ue_rxpower_data)
    energyload_metric = energyload_metric_generator(trafficload_ue_data, _site_config_data, ue_rxpower_data)

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

if __name__ == "__main__":

    site_config_data = pd.DataFrame(
        { #Create site_config_data
            c.CELL_ID: range(10),
            c.CELL_LAT: np.random.uniform(40.7, 40.8, 10),
            c.CELL_LON: np.random.uniform(-74.05, -73.95, 10),
            c.CELL_AZ_DEG: np.random.randint(0, 360, 10),
            c.CELL_EL_DEG: np.random.randint(0, 15, 10),
            c.CELL_TXPWR_DBM: np.random.uniform(20, 30, 10),
            c.HTX: [20] * 10,
            c.HRX: [2] * 10,
            c.CELL_CARRIER_FREQ_MHZ: [1800] * 10,
        }
    )

    spatial_params = {
        "types": ["residential", "commercial", "park"],
        "proportions": [0.5, 0.3, 0.2],
    }

    with open("spatial_params.json", "w") as f:
        json.dump(spatial_params, f)

    time_params = {
        "total_ticks": 24,
        "tick_duration": 1,
        "time_weights": {
            "_space1": [
                0.8,
                0.8,
                0.9,
                0.9,
                0.8,
                0.7,
                0.5,
                0.4,
                0.3,
                0.3,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.9,
                0.9,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
            ],
            "_space2": [
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.3,
                0.5,
                0.7,
                0.8,
                0.9,
                0.9,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
                0.3,
                0.3,
                0.3,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
            ],
            "_space3": [
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.2,
                0.3,
                0.4,
                0.4,
                0.4,
                0.3,
                0.3,
                0.3,
                0.2,
                0.2,
                0.2,
                0.2,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
            ],
        },
    }

    with open("time_params.json", "w") as f:
        json.dump(time_params, f)
    site_config_path = "site_config.csv"
    site_config_data.to_csv(site_config_path, index=False)

    total_ue = 500
    results = run_tl_simulation(site_config_path, total_ue, "spatial_params.json", "time_params.json")



    # --- Print Per-Tick Metrics ---
    print(
        "Traffic Load Metric (Std Dev of UE Counts per Tick):",
        results["trafficload_metric"],
    )
    print(
        "Energy Load Metric (Proportion of Cells Off per Tick):",
        results["energyload_metric"],
    )

    tick_to_plot = 0
    ue_data_tick = results["trafficload_ue_data"][
        results["trafficload_ue_data"]["tick"] == tick_to_plot
    ]
    ue_rxpower_data_tick = results["ue_rxpower_data"][
        results["ue_rxpower_data"]["tick"] == tick_to_plot
    ]

    ue_rxpower_data_tick = ue_rxpower_data_tick.loc[
        ue_rxpower_data_tick.groupby(["ue_id"])["rx_power"].idxmax()
    ] #No need of tick
    ue_data_tick = pd.merge(
        ue_data_tick,
        ue_rxpower_data_tick[["ue_id", c.CELL_ID]],
        left_on=["ue_id"], #Use only ue_id
        right_on=["ue_id"],
        how="left",
    )

    ue_data_tick.rename( #Correct renaming
        columns={f"{c.CELL_ID}_y": "serving_cell_id", f"{c.CELL_ID}_x": c.CELL_ID},
        inplace=True,
    )


    points = site_config_data[[c.CELL_LON, c.CELL_LAT]].values
    vor = Voronoi(points)
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        voronoi_plot_2d(
            vor,
            ax=ax,
            show_vertices=False,
            line_colors="gray",
            line_width=1,
            line_alpha=0.6,
            point_size=0,
        )
    except ImportError:
        print("scipy.spatial.voronoi_plot_2d not available, skipping Voronoi plot")


    ax.scatter(
        site_config_data[c.CELL_LON],
        site_config_data[c.CELL_LAT],
        marker="^",
        color="red",
        label="Cell Towers",
        s=50,
    )


    cmap = plt.get_cmap("tab20", len(site_config_data[c.CELL_ID]))

    for i, cell_id in enumerate(site_config_data[c.CELL_ID]):
        cell_ues = ue_data_tick[ue_data_tick["serving_cell_id"] == cell_id]
        if not cell_ues.empty:
            ax.scatter(
                cell_ues[c.LON],
                cell_ues[c.LAT],
                color=cmap(i),
                label=f"UEs (Cell {cell_id})",
                alpha=0.7,
                s=20,
            )


    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Voronoi Diagram with UEs (Tick {tick_to_plot}) - Colored by Serving Cell"
    )
    ax.legend()
    ax.grid(True)
    plt.savefig("./my_plot.png") #This line should create image in directory
    plt.close(fig)

