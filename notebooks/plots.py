import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from radp.digital_twin.utils.constants import LATENT_BACKGROUND_NOISE_DB, RLF_THRESHOLD


def plot_naive_attached_df(df: pd.DataFrame, topology: pd.DataFrame):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    unique_cells = topology["cell_id"].unique()

    predefined_colors = [
        (0.8, 0.0, 0.0),  # Dark Red
        (0.0, 0.6, 0.0),  # Dark Green
        (0.0, 0.0, 0.8),  # Dark Blue
        (0.0, 0.6, 0.6),  # Teal
        (0.7, 0.0, 0.7),  # Purple
        (0.8, 0.6, 0.0),  # Mustard
        (0.9, 0.3, 0.0),  # Orange-red
        (0.3, 0.0, 0.6),  # Indigo
        (0.3, 0.3, 0.3),  # Dark Gray
        (0.1, 0.1, 0.1),  # Almost black
    ]

    color_map = {cell_id: predefined_colors[i % len(predefined_colors)] for i, cell_id in enumerate(unique_cells)}

    # Flag to control RLF legend entry added once
    rlf_legend_added = False

    for cell_id in unique_cells:
        cell_data = topology[topology["cell_id"] == cell_id]
        lon = cell_data["cell_lon"].values[0]
        lat = cell_data["cell_lat"].values[0]
        color = color_map[cell_id]

        user_points = df[df["cell_id"] == cell_id]

        good_sinr_points = user_points[user_points["sinr_db"] >= RLF_THRESHOLD]
        bad_sinr_points = user_points[user_points["sinr_db"] < RLF_THRESHOLD]

        # Plot good SINR points with cell color and label
        if len(good_sinr_points) > 0:
            ax.scatter(
                good_sinr_points["loc_x"],
                good_sinr_points["loc_y"],
                color=[color] * len(good_sinr_points),
                alpha=0.6,
                s=40,
                marker="o",
                label=f"Users of Cell {cell_id}",
            )

        # Plot bad SINR points in grey, add label only once
        if len(bad_sinr_points) > 0:
            if not rlf_legend_added:
                ax.scatter(
                    bad_sinr_points["loc_x"],
                    bad_sinr_points["loc_y"],
                    color="grey",
                    alpha=0.6,
                    s=40,
                    marker="o",
                    label="RLF",
                )
                rlf_legend_added = True
            else:
                ax.scatter(
                    bad_sinr_points["loc_x"],
                    bad_sinr_points["loc_y"],
                    color="grey",
                    alpha=0.6,
                    s=40,
                    marker="o",
                    label=None,  # No repeated label
                )

        # Draw bold 'X' manually with two crossing thick lines + outline
        size = 0.005
        lw_main = 5
        lw_outline = 9

        ax.add_line(
            Line2D([lon - size, lon + size], [lat - size, lat + size], linewidth=lw_outline, color="black", zorder=2)
        )
        ax.add_line(
            Line2D([lon - size, lon + size], [lat + size, lat - size], linewidth=lw_outline, color="black", zorder=2)
        )

        ax.add_line(
            Line2D([lon - size, lon + size], [lat - size, lat + size], linewidth=lw_main, color=color, zorder=3)
        )
        ax.add_line(
            Line2D([lon - size, lon + size], [lat + size, lat - size], linewidth=lw_main, color=color, zorder=3)
        )

    ax.set_xlabel("Longitude / loc_x")
    ax.set_ylabel("Latitude / loc_y")
    ax.set_title("Bold Cell Centers with Strong X Outline & Colored Users")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def _compute_row_level_sinr(signal_dbm: float, interference_dbm_list: list, noise_db: float) -> float:
    signal_linear = 10 ** (signal_dbm / 10)
    interference_linear = sum(10 ** (p / 10) for p in interference_dbm_list)
    noise_linear = 10 ** (noise_db / 10)

    sinr_linear = signal_linear / (interference_linear + noise_linear)
    return 10 * np.log10(sinr_linear)


def add_sinr_db(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sinr_column = []

    # Group by location
    for (_, group) in df.groupby(["ue_id", "tick"]):
        # Group further by frequency layer within the same location
        freq_groups = group.groupby("cell_carrier_freq_mhz")

        # Create a temporary Series to store sinr values for current group
        group_sinr_values = pd.Series(index=group.index, dtype=float)

        for freq, freq_group in freq_groups:
            # List of all rx powers in this frequency group
            all_rxpowers = freq_group["cell_rxpower_dbm"].tolist()
            noise_db = LATENT_BACKGROUND_NOISE_DB

            for idx, row in freq_group.iterrows():
                serving_power = row["cell_rxpower_dbm"]
                # Remove this row's signal from interference
                interference_others = [p for p in all_rxpowers if p != serving_power or all_rxpowers.count(p) > 1]
                sinr_db = _compute_row_level_sinr(serving_power, interference_others, noise_db)
                group_sinr_values.at[idx] = sinr_db

        sinr_column.append(group_sinr_values)

    # Combine all the sinr values and add to DataFrame
    df["sinr_db"] = pd.concat(sinr_column).sort_index()

    return df
