import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi  # TODO: Check if scipy added to requirements
from shapely.geometry import Point, Polygon, box  # TODO: Check same for shapely
from shapely.validation import make_valid

from radp.digital_twin.utils import constants as c

# * --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _space_boundary(cell_topology_data: pd.DataFrame, buffer_percent=0.3) -> Dict:
    """Calculates the buffered bounding box."""
    if cell_topology_data.empty:
        raise ValueError("Cell topology data cannot be empty.")

    min_lat = cell_topology_data[c.CELL_LAT].min()
    max_lat = cell_topology_data[c.CELL_LAT].max()

    min_lon = cell_topology_data[c.CELL_LON].min()
    max_lon = cell_topology_data[c.CELL_LON].max()

    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    if lat_range > 1e-9:
        lat_buffer = lat_range * buffer_percent
    else:
        lat_buffer = 0.1

    if lon_range > 1e-9:
        lon_buffer = lon_range * buffer_percent
    else:
        lon_buffer = 0.1

    logger.debug(f"Original Bounds: LON=[{min_lon:.4f}, {max_lon:.4f}], LAT=[{min_lat:.4f}, {max_lat:.4f}]")
    logger.debug(f"Buffer Percent: {buffer_percent*100}%, Lon Buffer: {lon_buffer:.4f}, Lat Buffer: {lat_buffer:.4f}")

    return {
        "min_lon_buffered": max(min_lon - lon_buffer, -180.0),
        "min_lat_buffered": max(min_lat - lat_buffer, -90.0),
        "max_lon_buffered": min(max_lon + lon_buffer, 180.0),
        "max_lat_buffered": min(max_lat + lat_buffer, 90.0),
    }


def _space_component_list_generator(spatial_params: Dict) -> Dict:
    """Generates dict mapping space type names to proportions."""
    space_types = spatial_params.get("types", [])
    proportions = spatial_params.get("proportions", [])

    if not space_types or not proportions:
        raise ValueError("Spatial params need 'types' and 'proportions'.")

    if len(space_types) != len(proportions):
        raise ValueError("Num types and proportions must match.")

    if not np.isclose(sum(proportions), 1.0):
        raise ValueError("Proportions must sum to 1.0")

    # *** USE TYPE NAMES AS KEYS ***
    space_list = {stype: prop for stype, prop in zip(space_types, proportions)}

    logger.debug(f"Generated space component list: {space_list}")

    return space_list


def _create_spatial_cells(cell_topology_data: pd.DataFrame, space_list: Dict, space_bound: Dict) -> List[Dict]:
    """Creates clipped Voronoi cells with assigned space types (using actual type names)."""
    if cell_topology_data.empty:
        logger.error("Topology empty.")
        return []

    logger.info(f"Creating spatial cells from {len(cell_topology_data)} points.")

    if len(cell_topology_data) < 4:
        logger.error(f"Need >= 4 points for Voronoi, have {len(cell_topology_data)}.")
        return []

    if Voronoi is None:
        logger.error("Scipy required for Voronoi.")
        return []

    points = cell_topology_data[[c.CELL_LON, c.CELL_LAT]].values

    try:
        vor = Voronoi(points)
    except Exception as e:
        logger.error(f"Voronoi failed: {e}")
        return []

    spatial_cells = []

    min_lon = space_bound.get("min_lon_buffered")
    max_lon = space_bound.get("max_lon_buffered")
    min_lat = space_bound.get("min_lat_buffered")
    max_lat = space_bound.get("max_lat_buffered")

    if any(v is None for v in [min_lon, max_lon, min_lat, max_lat]):
        logger.error("Invalid space boundary dict.")
        return []

    logger.debug(f"Clipping boundary: LON=[{min_lon:.4f}, {max_lon:.4f}], LAT=[{min_lat:.4f}, {max_lat:.4f}]")

    try:
        boundary_polygon = box(min_lon, min_lat, max_lon, max_lat)  # type: ignore
        assert boundary_polygon.is_valid
    except Exception as e:
        logger.error(f"Failed boundary polygon creation: {e}")
        return []

    sk_inf, sk_inv, sk_emp, proc = (0, 0, 0, 0)

    for i, region_idx_list in enumerate(vor.regions):
        if not region_idx_list or -1 in region_idx_list:
            sk_inf += 1
            continue

        try:
            verts = vor.vertices[region_idx_list]

            if np.isnan(verts).any():
                logger.warning(f"Skip region {i}: NaN vertices.")
                sk_inv += 1
                continue

            if len(verts) < 3:
                logger.warning(f"Skip region {i}: <3 vertices.")
                sk_inv += 1
                continue

            vor_poly = Polygon(verts)

            if not vor_poly.is_valid:
                vor_poly = make_valid(vor_poly)

            if not vor_poly.is_valid:
                logger.warning(f"Skip region {i}: Invalid polygon.")
                sk_inv += 1
                continue

            polys_to_use = []

            if vor_poly.geom_type == "Polygon":
                polys_to_use.append(vor_poly)
            elif vor_poly.geom_type == "MultiPolygon":
                polys_to_use.extend(list(vor_poly.geoms))  # type: ignore
            else:
                logger.warning(f"Skip region {i}: make_valid gave {vor_poly.geom_type}")
                sk_inv += 1
                continue

            valid_clipped_polys = []

            for p in polys_to_use:
                if p.geom_type != "Polygon":
                    continue

                clipped = p.intersection(boundary_polygon)

                if not clipped.is_empty and clipped.is_valid:
                    if clipped.geom_type == "Polygon" and clipped.area > 1e-9:
                        valid_clipped_polys.append(clipped)
                    elif clipped.geom_type == "MultiPolygon":
                        for sub_poly in clipped.geoms:
                            if sub_poly.geom_type == "Polygon" and sub_poly.is_valid and sub_poly.area > 1e-9:
                                valid_clipped_polys.append(sub_poly)

            if not valid_clipped_polys:
                sk_emp += 1
                continue

            space_keys = list(space_list.keys())
            space_probs = list(space_list.values())

            if not np.isclose(sum(space_probs), 1.0):
                space_probs = np.array(space_probs) / sum(space_probs)

            assigned_type = np.random.choice(space_keys, p=space_probs)

            for final_poly in valid_clipped_polys:
                spatial_cells.append(
                    {
                        "bounds": list(final_poly.exterior.coords),
                        "type": assigned_type,
                        "cell_id": len(spatial_cells),
                        "original_voronoi_region": i,
                    }
                )

                proc += 1

        except IndexError:
            logger.warning(f"IndexError region {i}.")
            sk_inv += 1
            continue
        except Exception as e:
            logger.error(f"Error region {i}: {e}.")
            sk_inv += 1
            continue

    logger.info(
        f"""Voronoi processing: Processed OK={proc}, Skip Inf={sk_inf}, Skip Invalid={sk_inv}, Skip EmptyClip={sk_emp}.
        Total cells={len(spatial_cells)}"""
    )

    if not spatial_cells:
        logger.error("No valid spatial cells created after clipping.")

    return spatial_cells


def city_digitaltwin_generator(
    total_ue: int, spatial_params_for_city: Dict, time_params: Dict, cell_topology_data: pd.DataFrame
) -> pd.DataFrame:
    """Generates UE locations using actual type names."""
    logger.info(f"Generating UE data for {total_ue} UEs over {time_params.get('total_ticks', 0)} ticks.")

    space_bound = _space_boundary(cell_topology_data)
    space_list = _space_component_list_generator(spatial_params_for_city)  # Now returns {'residential': 0.5, ...}
    spatial_cells = _create_spatial_cells(cell_topology_data, space_list, space_bound)

    if not spatial_cells:
        logger.error("No spatial cells; cannot generate UEs.")
        return pd.DataFrame()

    total_ticks = time_params.get("total_ticks", 1)
    ue_data = []
    time_weights = time_params.get("time_weights", {})

    COL_LAT = getattr(c, "LAT", "lat")
    COL_LON = getattr(c, "LON", "lon")

    for tick in range(total_ticks):
        tick_weights = []
        valid_cells_for_tick = []

        for cell in spatial_cells:
            cell_type = cell["type"]  # This is now 'residential', etc.

            # Lookup in time_weights uses actual type names
            type_weights = time_weights.get(cell_type, [1.0] * total_ticks)

            if tick < len(type_weights):
                weight = type_weights[tick]
                if weight > 0:
                    tick_weights.append(weight)
                    valid_cells_for_tick.append(cell)

        if not tick_weights or np.sum(tick_weights) == 0:
            logger.warning(f"No valid cells/weights for tick {tick}. Skipping.")
            continue

        probs = np.array(tick_weights) / np.sum(tick_weights)

        for ue_id in range(total_ue):
            chosen_cell = np.random.choice(valid_cells_for_tick, p=probs)

            try:
                polygon = Polygon(chosen_cell["bounds"])
                min_lon, min_lat, max_lon, max_lat = polygon.bounds

                attempts = 0
                max_attempts = 100

                while attempts < max_attempts:
                    ue_lon = np.random.uniform(min_lon, max_lon)
                    ue_lat = np.random.uniform(min_lat, max_lat)
                    point = Point(ue_lon, ue_lat)

                    if polygon.contains(point):
                        break

                    attempts += 1

                else:
                    logger.warning(
                        f"Point generation failed for cell {chosen_cell['cell_id']}, UE {ue_id}, tick {tick}. Skipping."
                    )
                    continue

                ue_data.append(
                    {
                        "tick": tick,
                        "ue_id": ue_id,
                        COL_LAT: ue_lat,
                        COL_LON: ue_lon,
                        "space_type": chosen_cell["type"],  # Store actual type name
                        "voronoi_cell_id": chosen_cell["cell_id"],
                    }
                )

            except Exception as e:
                logger.error(f"Error placing UE {ue_id} tick {tick}: {e}")
                continue

    if not ue_data:
        logger.warning("No UE data generated.")
        return pd.DataFrame()

    logger.info(f"Generated {len(ue_data)} UE data points.")
    return pd.DataFrame(ue_data)
