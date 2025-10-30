import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from radp.digital_twin.utils.constants import LATENT_BACKGROUND_NOISE_DB, RLF_THRESHOLD


def plot_sinr_db_by_ue(
    df: pd.DataFrame, df2: pd.DataFrame, ue_id: int, save_path: str, rlf_threshold=RLF_THRESHOLD
) -> None:
    """
    Plots SINR (in dB) over ticks for a specific ue_id and saves to file.

    - Solid bold line: Connected cell_id (from df), color-coded.
    - Dotted lines: All cell_id sinr_db values from df2 for context.
    - RLF events: Drop to bottom with bold black line.
    - RLF_THRESHOLD: Horizontal dashed line.

    Parameters:
    df (pd.DataFrame): Connected cell data: 'ue_id', 'tick', 'sinr_db', 'cell_id' (or 'RLF').
    df2 (pd.DataFrame): All candidate cell data: 'ue_id', 'tick', 'cell_id', 'sinr_db'.
    ue_id (int): UE to plot.
    save_path (str): File path where the plot will be saved (e.g., '/path/to/plot.png').

    +--------+------+----------+----------+
    | ue_id  | tick | cell_id  | sinr_db  |
    +========+======+==========+==========+
    |   0    |  0   |    1     |  14.0    |
    |   0    |  0   |    2     |  12.5    |
    |   1    |  0   |    1     |  13.2    |
    |   1    |  0   |    2     |  10.8    |
    |   0    |  1   |    1     |  16.7    |
    |   0    |  1   |    2     |  12.3    |
    |   1    |  1   |    1     |  -2.0    |
    |   1    |  1   |    2     |  -4.3    |
    +--------+------+----------+----------+

    """
    ue_df = df[df["ue_id"] == ue_id].sort_values("tick").reset_index(drop=True)
    ue_df2 = df2[df2["ue_id"] == ue_id].sort_values("tick")

    if ue_df.empty or ue_df2.empty:
        print(f"No data found for ue_id {ue_id}.")
        return

    # Base + dynamic color map
    base_colors = {1.0: "red", 2.0: "green", 3.0: "blue"}
    all_cell_ids = pd.concat([ue_df2["cell_id"], ue_df[ue_df["cell_id"] != "RLF"]["cell_id"]]).unique()
    missing_ids = [cid for cid in all_cell_ids if cid not in base_colors]
    extra_colors = cm.get_cmap("tab10", len(missing_ids))
    dynamic_colors = {cid: extra_colors(i) for i, cid in enumerate(missing_ids)}
    full_color_map = {**base_colors, **dynamic_colors}

    min_sinr = min(ue_df2["sinr_db"].min(), ue_df[ue_df["cell_id"] != "RLF"]["sinr_db"].min())
    drop_value = min_sinr - 5

    plt.figure(figsize=(12, 6))
    legend_cells = set()

    # --- Plot all candidate cell SINRs (dotted, bold) ---
    for cell_id, group in ue_df2.groupby("cell_id"):
        label = f"cell_id {cell_id}" if cell_id not in legend_cells else None
        legend_cells.add(cell_id)
        plt.plot(
            group["tick"],
            group["sinr_db"],
            linestyle=":",
            linewidth=2.5,
            color=full_color_map.get(cell_id, "gray"),
            label=label,
            alpha=0.7,
        )

    # --- Plot connected UE SINR as a continuous line, color-coded per cell_id ---
    for i in range(len(ue_df) - 1):
        tick1, tick2 = ue_df.loc[i, "tick"], ue_df.loc[i + 1, "tick"]
        sinr1, sinr2 = ue_df.loc[i, "sinr_db"], ue_df.loc[i + 1, "sinr_db"]
        cell1, cell2 = ue_df.loc[i, "cell_id"], ue_df.loc[i + 1, "cell_id"]

        # If current or next point is RLF, break the line
        if cell1 == "RLF" or cell2 == "RLF":
            continue

        # Draw line from point i to i+1 with color of current cell
        label = f"cell_id {cell1}" if cell1 not in legend_cells else None
        if label:
            legend_cells.add(cell1)
        plt.plot(
            [tick1, tick2],
            [sinr1, sinr2],
            color=full_color_map.get(cell1, "gray"),
            linewidth=3,
            label=label,
        )

    # --- Plot RLFs as vertical drops ---
    rlf_ticks = ue_df[ue_df["cell_id"] == "RLF"]["tick"]
    if not rlf_ticks.empty:
        for rlf_tick in rlf_ticks:
            plt.plot([rlf_tick], [drop_value], "ko", markersize=8, label="RLF" if "RLF" not in legend_cells else None)
            legend_cells.add("RLF")

    # --- RLF Threshold ---
    plt.axhline(y=rlf_threshold, color="black", linestyle="--", linewidth=2)

    # --- Final Decorations ---
    plt.title(f"SINR over Time for UE ID {ue_id}")
    plt.xlabel("Tick")
    plt.ylabel("SINR (dB)")
    plt.grid(True)
    plt.legend(title=None, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def mro_plot_scatter(df: pd.DataFrame, topology: pd.DataFrame, save_path: str, rlf_threshold=RLF_THRESHOLD) -> None:
    """
    Plot a scatter plot of cell towers and UE (User Equipment) locations and saves to file.
    @param df: DataFrame containing UE data with columns 'loc_x', 'loc_y', 'cell_id', and 'sinr_db'.
    @param topology: DataFrame containing cell tower data with columns 'cell_lon', 'cell_lat', and 'cell_id'.
    @param save_path: File path where the plot will be saved (e.g., '/path/to/plot.png').
    @returns: None. Saves a scatter plot with cell towers and UE locations.

    df:
    +---------+--------+--------+----------+
    | cell_id | loc_x  | loc_y  | sinr_db  |
    +=========+========+========+==========+
    |    1    | 90.412 | 23.810 |   15.3   |
    |    2    | 90.413 | 23.811 |   12.1   |
    |    1    | 90.415 | 23.812 |   18.7   |
    |    2    | 90.416 | 23.813 |    5.5   |
    +---------+--------+--------+----------+

    topology:
    +---------+----------+----------+
    | cell_id | cell_lon | cell_lat |
    +=========+==========+==========+
    |    1    | 90.410   | 23.809   |
    |    2    | 90.414   | 23.810   |
    +---------+----------+----------+

    """

    # Create a figure and axis
    plt.figure(figsize=(10, 8))

    plt.scatter([], [], color="grey", label="RLF")

    # Define color mapping based on cell_id for both cells and UEs (dynamic)
    base_colors = {1: "red", 2: "green", 3: "blue"}
    all_cell_ids = pd.concat([topology["cell_id"], df["cell_id"]]).unique()
    missing_ids = [cid for cid in all_cell_ids if cid not in base_colors]
    extra_colors = cm.get_cmap("tab10", len(missing_ids))
    dynamic_colors = {cid: extra_colors(i) for i, cid in enumerate(missing_ids)}
    color_map = {**base_colors, **dynamic_colors}

    # Plot cell towers from the topology dataframe with triangle markers and corresponding colors
    for _, row in topology.iterrows():
        color = color_map.get(row["cell_id"], "black")  # Default to black if unknown cell_id
        plt.scatter(
            row["cell_lon"],
            row["cell_lat"],
            marker="^",
            color=color,
            s=200,
            label=f"Cell {row['cell_id']}",
        )

    # Plot UEs from df without labels but with the same color coding
    for _, row in df.iterrows():
        color = color_map.get(row["cell_id"], "black")  # Default to black if unknown cell_id
        if row["sinr_db"] < rlf_threshold:  # REMOVE COMMENT WHEN sinr_db IS FIXED
            color = "grey"  # Change to grey if sinr_db < 2

        plt.scatter(row["loc_x"], row["loc_y"], color=color)

    # Add labels and title
    plt.xlabel("Longitude (loc_x)")
    plt.ylabel("Latitude (loc_y)")
    plt.title("Cell Towers and UE Locations")

    # Create a legend for the cells only
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def add_sinr_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'sinr_db' column to the input DataFrame, computing the Signal-to-Interference-plus-Noise Ratio (SINR)
    for each UE–cell pair based on received signal power, background noise, and interference.

    Parameters:
        df (pd.DataFrame): DataFrame with 'ue_id', 'cell_rxpower_dbm', and 'cell_carrier_freq_mhz' per row.

    Returns:
        pd.DataFrame: Updated DataFrame with an additional 'sinr_db' column.

    +--------+---------+------------------+------------------------+
    | ue_id  | cell_id | cell_rxpower_dbm | cell_carrier_freq_mhz  |
    +========+=========+==================+========================+
    |   0    |    1    |   -100.311970    |         2100.0         |
    |   0    |    2    |    -99.841523    |         2100.0         |
    |   1    |    1    |   -100.294405    |         2100.0         |
    |   1    |    2    |   -100.132420    |         2100.0         |
    |   2    |    1    |   -100.650003    |         2100.0         |
    |   2    |    2    |   -100.456381    |         2100.0         |
    |   3    |    1    |   -100.987321    |         2100.0         |
    |   3    |    2    |   -100.864529    |         2100.0         |
    +--------+---------+------------------+------------------------+

    """
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


# Compute SINR for each row (UE–cell pair), given its group
def _compute_row_level_sinr(signal_dbm: float, interference_dbm_list: list, noise_db: float) -> float:
    """
        Computes the SINR for a single UE–cell pair by removing interference
        and noise from the received signal power.

    Parameters:
            row (pd.Series): Current row containing signal data.
            group (pd.DataFrame): Group of UE–cell rows sharing the same UE and frequency.

        +--------+---------+------------------+------------------------+
        | ue_id  | cell_id | cell_rxpower_dbm | cell_carrier_freq_mhz |
        +========+=========+==================+========================+
        |   0    |    1    |   -100.311970    |         2100.0         |
        |   0    |    2    |    -99.841523    |         2100.0         |
        |   1    |    1    |   -100.294405    |         2100.0         |
        |   1    |    2    |   -100.132420    |         2100.0         |
        |   2    |    1    |   -100.650003    |         2100.0         |
        |   2    |    2    |   -100.456381    |         2100.0         |
        |   3    |    1    |   -100.987321    |         2100.0         |
        |   3    |    2    |   -100.864529    |         2100.0         |
        +--------+---------+------------------+------------------------+


        Returns:
            float: The computed SINR value in decibels for the current UE–cell pair.
    """
    signal_linear = 10 ** (signal_dbm / 10)
    interference_linear = sum(10 ** (p / 10) for p in interference_dbm_list)
    noise_linear = 10 ** (noise_db / 10)

    sinr_linear = signal_linear / (interference_linear + noise_linear)
    return 10 * np.log10(sinr_linear)
