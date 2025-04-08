import sys
import json

import pandas as pd
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib.cm as cm

RADP_ROOT = ""
sys.path.insert(0, RADP_ROOT)

from radp.digital_twin.utils import constants as c
from radp.digital_twin.traffic_load_model.traffic_load_model import run_tl_simulation


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


# * run traffic load simulation
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
    marker="^", # type: ignore
    color="red",
    label="Cell Towers",
    s=50,
)

cmap = cm.get_cmap("tab20", len(site_config_data[c.CELL_ID]))

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
