from typing import Dict, List

import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon
from radp.digital_twin.utils import constants as c


# --- Helper Functions ---

def _space_boundary(cell_topology_data: pd.DataFrame) -> Dict:
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

def _space_component_list_generator(spatial_params: Dict) -> Dict:
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


def city_digitaltwin_generator(
    total_ue: int,
    spatial_params_for_city: Dict,
    time_params: Dict,
    cell_topology_data: pd.DataFrame,
) -> pd.DataFrame:
    """Generates UE data over time, with improved UE distribution."""

    space_bound = _space_boundary(cell_topology_data)
    space_list = _space_component_list_generator(spatial_params_for_city)
    spatial_cells = _create_spatial_cells(cell_topology_data, space_list, space_bound)
    total_ticks = time_params["total_ticks"]
    # tick_duration = time_params["tick_duration"]
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
            chosen_index = np.random.choice(len(spatial_cells), p=probs)
            chosen_cell = spatial_cells[chosen_index]
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