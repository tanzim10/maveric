# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Default logic for cell selection, camping and attachment.
"""

from collections import defaultdict
from operator import itemgetter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from radp.digital_twin.utils import constants


def perform_attachment(
    ue_prediction_data: pd.DataFrame,
    topology: pd.DataFrame,
) -> pd.DataFrame:
    """This method looks at Rx power (predictions) from cells to pixels,
    for a given snapshot in time (e.g. simulation tick),
    and performs cell selection (camping or attachment).

    `ue_prediction_data` is a dataframe which contains at least the
    following columns:
        lon,            # pixel longitude or similar
        lat,            # pixel latitude or similar
        cell_id,        # cell ID for a cell
        rxpower_dbm,    # (predicted) Rx power from given cell to given location

    `topology` is a dataframe which contains at least the following columns:
        cell_id:                # cell ID for a cell
        cell_carrier_freq_mhz:  # carrier frequency (in MHz) for given cell

    This method returns a single dataframe back that represents
    the result of cell selection. The following columns are present in the result :
        loc_x,      # pixel longitude or similar
        loc_y,      # pixel latitude or similar
        cell_id,    # cell ID of selected cell
        rsrp_dbm,   # (predicted) Rx power of selected cell
        sinr_db,    # (predicted) SINR, that accounts for
                    # interference from other cells on the same layer (carrier frequency)
    """

    # initiate a dictionary to store power-by-layer dictionaries on a per-pixel basis
    rx_powers_by_layer_by_loc: Dict[Tuple[float, float], Dict[float, List[Tuple[Any, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # pull per-cell frequencies for faster lookup
    cell_id_to_freq = {row.cell_id: row.cell_carrier_freq_mhz for _, row in topology.iterrows()}

    # iterate over ue_prediction_data, to
    # build rx_powers_by_layer_by_loc map
    for _, row in ue_prediction_data.iterrows():
        # pull cell carrier frequency
        cell_carrier_freq_mhz = cell_id_to_freq[row.cell_id]

        # fetch the pixel longitude and latitude
        loc_x = row.get(constants.LOC_X, row.get(constants.LON))
        loc_y = row.get(constants.LOC_Y, row.get(constants.LAT))
        if loc_x is None or loc_y is None:
            raise Exception("loc_x or loc_y cannot be found in the dataset")

        # add (cell_id, rxpower) tuple on a per-row, per-freq basis
        rx_powers_by_layer_by_loc[(loc_x, loc_y)][cell_carrier_freq_mhz].append((row.cell_id, row.rxpower_dbm))

    # perform cell selection per location
    rf_dataframe_dict = defaultdict(list)

    for loc, rx_powers_by_layer in rx_powers_by_layer_by_loc.items():
        # compute strongest server, interference and SINR
        rsrp_dbm_by_layer, sinr_db_by_layer = get_rsrp_dbm_sinr_db_by_layer(rx_powers_by_layer)

        # pull sinr_db, cell_id and rsrp_dbm based on highest SINR
        max_sinr_db_item = max(sinr_db_by_layer.items(), key=lambda k: k[1][1])
        max_sinr_db_cell_id, max_sinr_db = max_sinr_db_item[1]
        rsrp_dbm = next(v[1] for v in rsrp_dbm_by_layer.values() if v[0] == max_sinr_db_cell_id)

        # update rf_dataframe output
        rf_dataframe_dict[constants.LOC_X].append(loc[0])
        rf_dataframe_dict[constants.LOC_Y].append(loc[1])
        rf_dataframe_dict[constants.CELL_ID].append(max_sinr_db_cell_id)
        rf_dataframe_dict[constants.SINR_DB].append(max_sinr_db)
        rf_dataframe_dict[constants.RSRP_DBM].append(rsrp_dbm)

    # return as dataframe
    return pd.DataFrame(rf_dataframe_dict)


def get_rsrp_dbm_sinr_db_by_layer(
    rx_powers_by_layer: Dict[float, List[Tuple[str, float]]],
) -> Tuple[Dict[float, Tuple[str, float]], Dict[float, Tuple[str, float]]]:
    """Given (predicted) Rx powers per layer, determine
    RSRP (served by max power) and SINR, within layer.
    """

    rsrp_dbm_by_layer: Dict[float, Tuple[str, float]] = {}
    sinr_db_by_layer: Dict[float, Tuple[str, float]] = {}

    # for each layer, compute strongest server, interference and SINR
    for cell_carrier_freq_mhz, rx_powers in rx_powers_by_layer.items():
        # get the max rsrp from the layer
        rsrp_dbm_by_layer[cell_carrier_freq_mhz] = max(rx_powers, key=itemgetter(1))

        # calculate background noise
        pred_noise = 10 ** (constants.LATENT_BACKGROUND_NOISE_DB / 10)

        # calculate prediction interference by subtracting
        # the strongest rx power from total sum of rx powers
        pred_interference = sum(10 ** (p / 10) for i, p in rx_powers) - (
            10 ** (rsrp_dbm_by_layer[cell_carrier_freq_mhz][1] / 10)
        )
        pred_interference_noise_dBm = 10 * np.log10(pred_interference + pred_noise)
        sinr_db_by_layer[cell_carrier_freq_mhz] = (
            rsrp_dbm_by_layer[cell_carrier_freq_mhz][0],
            (rsrp_dbm_by_layer[cell_carrier_freq_mhz][1] - pred_interference_noise_dBm),
        )

    return rsrp_dbm_by_layer, sinr_db_by_layer


def perform_attachment_hyst_ttt(ue_data: pd.DataFrame, hyst: float, ttt: int, rlf_threshold: float) -> pd.DataFrame:
    """
    Performs UE-to-cell attachment across all ticks in the simulation.
    Initially, when insufficient history is available, the UE is naively attached to the strongest available cell.
    Once the required Time-To-Trigger (TTT) history is built up, Hysteresis (Hyst) and TTT logic are applied to ensure
    more stable and realistic attachment decisions.

    Parameters:
        ue_data (pd.DataFrame): UE measurements with 'tick' and 'cell_rxpower_dbm'.
        hyst (float): Hysteresis threshold in dB.
        ttt (int): Time-to-trigger window size.
        rlf_threshold (float): Minimum signal level to maintain a connection.

    Returns:
        pd.DataFrame: DataFrame of all UE-cell attachment states across all ticks.
    """
    strongest_server_history = []
    current_attachment = pd.DataFrame()

    tick_dataframes = {}
    # Group the data by tick
    tick_dataframes = {tick: group.copy() for tick, group in ue_data.groupby("tick")}

    cell_attached_df = []
    for tick in sorted(tick_dataframes.keys()):
        use_strongest_server = len(strongest_server_history) < ttt - 1
        strongest_server_history, current_attachment = _perform_attachment_hyst_ttt_per_tick(
            tick_dataframes[tick],
            strongest_server_history,
            current_attachment,
            ttt,
            hyst,
            use_strongest_server=use_strongest_server,
        )
        current_attachment = _check_rlf_threshold(current_attachment, tick_dataframes[tick], rlf_threshold)
        cell_attached_df.append(current_attachment)

    cell_attached_df = pd.concat(cell_attached_df, ignore_index=True)

    return cell_attached_df


def find_hyst_diff(df2: pd.DataFrame) -> float:
    """
    Finds the highest difference in the 'cell_rxpower_dbm' column
    by calculating the absolute difference between consecutive rows.

    Parameters:
        df2 (pd.DataFrame): Input DataFrame containing the 'cell_rxpower_dbm' column.

    Returns:
        float: The maximum absolute difference between consecutive values
    """
    # Make a copy of the dataframe to avoid modifying the original one
    df = df2.copy()

    # Replace infinite values with NaN in 'cell_rxpower_dbm'
    df["cell_rxpower_dbm"] = df["cell_rxpower_dbm"].replace([np.inf, -np.inf], np.nan)

    # Drop rows where 'cell_rxpower_dbm' is NaN
    df_clean = df.dropna(subset=["cell_rxpower_dbm"]).copy()

    # Calculate the difference between the maximum and minimum values in the cleaned data
    max_val = df_clean["cell_rxpower_dbm"].max()
    min_val = df_clean["cell_rxpower_dbm"].min()

    # Return the difference between the max and min values
    return max_val - min_val


def _perform_attachment_hyst_ttt_per_tick(
    ue_data_for_current_tick: pd.DataFrame,
    strongest_server_history: List[pd.DataFrame],
    past_attachment: pd.DataFrame,
    ttt: int,
    hyst: float,
    use_strongest_server: bool = False,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Determines and updates the cell selction for each user for a given tick using
    hysteresis and time-to-trigger (TTT) rules.
    This function either attaches the UE to the strongest server or evaluates
    attachment changes based on hyst
    and TTT rules. The attachment decisions are updated, and the history of the strongest server
    is maintained across ticks.

    Parameters:
        ue_data_for_current_tick (pd.DataFrame): UE data for the current tick containing measurements like
                                                  `cell_rxpower_dbm` for each UE.
        strongest_server_history (List[pd.DataFrame]): A history of the strongest server attachments for previous ticks.
        past_attachment (pd.DataFrame): Attachment state of
        the UE from the previous tick.
        ttt (int): Time-to-trigger (TTT) threshold for attachment decision, in number of ticks.
        hyst (float): Hysteresis threshold in dB for attachment decision.
        use_strongest_server (bool): If True, forces attachment to the strongest server;
        otherwise, follows TTT and hyst rules.


    Returns:
        Tuple[List[pd.DataFrame], pd.DataFrame]:
            - Updated list of the strongest server history (List of DataFrames).
            - Current UE-cell attachment decision (DataFrame).
    """
    # Determine the strongest server for the current tick
    current_strongest = ue_data_for_current_tick.loc[
        ue_data_for_current_tick.groupby("ue_id")["cell_rxpower_dbm"].idxmax()
    ]

    # Validate strongest_server_history length
    if len(strongest_server_history) >= ttt:
        raise ValueError("Strongest_Server_History length must be less than TTT!")

    # Use strongest server directly if specified
    if use_strongest_server:
        strongest_server_history.append(current_strongest)
        current_attachment = current_strongest
    else:
        # Ensure strongest_server_history length matches TTT - 1
        if len(strongest_server_history) + 1 == ttt:
            # Apply hysteresis check
            current_strongest = _check_hyst(ue_data_for_current_tick, past_attachment, hyst)
            strongest_server_history.append(current_strongest)

            # Apply TTT logic
            current_attachment = _check_ttt(strongest_server_history, ue_data_for_current_tick, past_attachment)

            # Final hysteresis check within the current tick
            current_attachment = _check_hyst_in_current_tick(
                ue_data_for_current_tick, current_attachment, past_attachment, hyst
            )
        else:
            raise ValueError(
                "Strongest_Server_History length must equal TTT - 1. "
                "Call Perform Attachment with use_strongest_server=True."
            )

    # Maintain history length within TTT
    if len(strongest_server_history) == ttt:
        strongest_server_history.pop(0)

    return strongest_server_history, current_attachment


def _check_hyst(ue_data_for_current_tick: pd.DataFrame, past_attachment: pd.DataFrame, hyst: float) -> pd.DataFrame:
    """
    Evaluates if a newly strongest cell is significantly better than the previously attached cell
    by applying the hysteresis (hyst) margin.

    Hyst: Ensures the new strongest cell is *significantly* better than the currently attached one
    (i.e., current_rxpower - past_rxpower ≥ hyst) to prevent unnecessary handovers.

    Parameters:
        ue_data_for_current_tick (pd.DataFrame): Current tick signal data containing
        'ue_id', 'cell_id', and 'cell_rxpower_dbm'.
        past_attachment (pd.DataFrame): Previous attachment decisions per UE.
        hyst (float): Hysteresis threshold to validate signal superiority.

    +--------------------------------------------+
    | ue_data_for_current_tick                   |
    +--------+---------+-------------------------+
    | ue_id  | cell_id | cell_rxpower_dbm | tick |
    +========+=========+=========================+
    |   0    |    1    |   -100.311970    |  1   |
    |   0    |    2    |    -99.841523    |  1   |
    |   1    |    1    |   -100.294405    |  1   |
    |   1    |    2    |   -100.132420    |  1   |
    |   2    |    1    |   -100.650003    |  1   |
    |   2    |    2    |   -100.456381    |  1   |
    +--------+---------+-------------------------+

    +--------------------------------------------+
    | past_attachment                            |
    +--------+---------+-------------------------+
    | ue_id  | cell_id | cell_rxpower_dbm | tick |
    +========+=========+=========================+
    |   0    |    1    |   -100.723849    |  0   |
    |   0    |    2    |    -99.841523    |  0   |
    |   1    |    1    |   -100.933915    |  0   |
    |   1    |    2    |   -100.132420    |  0   |
    |   2    |    1    |    -99.298523    |  0   |
    |   2    |    2    |   -100.122649    |  0   |
    +--------+---------+-------------------------+

    Returns:
        pd.DataFrame: Updated attachment decisions after applying the hysteresis condition.

    """
    # Merge the current tick data with past attachment data (to compare past power)
    merged_df = pd.merge(
        ue_data_for_current_tick,
        past_attachment,
        on="ue_id",
        how="left",
        suffixes=("", "_past"),
    )

    # Initialize an empty dictionary to store the best rows for each UE
    best_rows = {}

    # Iterate through the merged DataFrame
    for _, row in merged_df.iterrows():
        ue_id = row["ue_id"]
        current_power = row["cell_rxpower_dbm"]
        past_cell_id = row["cell_id_past"]

        # Retrieve the past power for the previous cell_id of this ue_id
        past_power = (
            ue_data_for_current_tick.loc[
                (ue_data_for_current_tick["ue_id"] == ue_id) & (ue_data_for_current_tick["cell_id"] == past_cell_id),
                "cell_rxpower_dbm",
            ].max()
            if past_cell_id in ue_data_for_current_tick["cell_id"].values
            else -999
        )

        # Determine the best row based on hysteresis and power comparison
        if current_power - past_power >= hyst:
            if ue_id not in best_rows or current_power > best_rows[ue_id]["cell_rxpower_dbm"]:
                best_rows[ue_id] = row
        elif past_power > current_power:
            if ue_id not in best_rows or past_power > best_rows[ue_id]["cell_rxpower_dbm"]:
                best_rows[ue_id] = ue_data_for_current_tick.loc[
                    (ue_data_for_current_tick["ue_id"] == ue_id) & (ue_data_for_current_tick["cell_id"] == past_cell_id)
                ].iloc[0]

    # Convert the dictionary of best rows into a DataFrame
    final_df = pd.DataFrame(best_rows.values())

    # Remove columns that have the '_past' suffix (since we only need current data)
    final_df = final_df.loc[:, ~final_df.columns.str.endswith("_past")]

    return final_df


def _check_ttt(
    strongest_server_history: List[pd.DataFrame],
    ue_data_for_current_tick: pd.DataFrame,
    past_attachment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensures that a UE only switches to a new cell if it has consistently been
    the strongest cell for a full Time-To-Trigger (TTT) duration.

    TTT: Ensures signal superiority is sustained over time by only allowing a handover if the same cell has remained
    the strongest for the full TTT duration.

    Parameters:
        strongest_server_history (list): List of DataFrames tracking the strongest
        cell per UE for past ticks (length = TTT - 1).
        ue_data_for_current_tick (pd.DataFrame): DataFrame containing current tick’s UE–cell signal data.
        past_attachment (pd.DataFrame): DataFrame with previous UE–cell attachment state.

        +--------------------------------------------+
        | ue_data_for_current_tick                   |
        +--------+---------+-------------------------+
        | ue_id  | cell_id | cell_rxpower_dbm | tick |
        +========+=========+=========================+
        |   0    |    1    |   -100.311970    |  1   |
        |   0    |    2    |    -99.841523    |  1   |
        |   1    |    1    |   -100.294405    |  1   |
        |   1    |    2    |   -100.132420    |  1   |
        |   2    |    1    |   -100.650003    |  1   |
        |   2    |    2    |   -100.456381    |  1   |
        +--------+---------+-------------------------+

        +--------------------------------------------+
        | past_attachment                            |
        +--------+---------+-------------------------+
        | ue_id  | cell_id | cell_rxpower_dbm | tick |
        +========+=========+=========================+
        |   0    |    1    |   -100.723849    |  0   |
        |   0    |    2    |    -99.841523    |  0   |
        |   1    |    1    |   -100.933915    |  0   |
        |   1    |    2    |   -100.132420    |  0   |
        |   2    |    1    |    -99.298523    |  0   |
        |   2    |    2    |   -100.122649    |  0   |
        +--------+---------+-------------------------+

    Returns:
        pd.DataFrame: Updated UE–cell attachment decisions after applying the TTT rule.
    """
    # Combine strongest server history into a single DataFrame
    merged_df = pd.concat(strongest_server_history, ignore_index=True)

    # Create a dictionary for quick lookup of past cell attachments
    past_attachment_dict = past_attachment.set_index("ue_id")["cell_id"].to_dict()

    # Initialize list for updated UE-cell attachments
    current_attachment_list = []

    # Iterate over unique UE IDs in the current tick data
    for ue_id in ue_data_for_current_tick["ue_id"].unique():
        # Filter merged history for the current UE
        ue_history = merged_df[merged_df["ue_id"] == ue_id]

        # Check if the cell attachment history is consistent
        unique_cells = ue_history["cell_id"].unique()
        if len(unique_cells) == 1:
            # Attach to the consistent cell
            consistent_cell_id = unique_cells[0]
            current_attachment_list.append(
                ue_data_for_current_tick.loc[
                    (ue_data_for_current_tick["ue_id"] == ue_id)
                    & (ue_data_for_current_tick["cell_id"] == consistent_cell_id)
                ].iloc[0]
            )
        else:
            # Get the past cell attachment for the current UE
            past_cell_id = past_attachment_dict.get(ue_id, "RLF")
            if past_cell_id == "RLF":
                # Attach to the cell with the highest rxpower
                current_attachment_list.append(
                    ue_data_for_current_tick.loc[ue_data_for_current_tick["ue_id"] == ue_id]
                    .nlargest(1, "cell_rxpower_dbm")
                    .iloc[0]
                )
            else:
                # Attach to the past cell
                current_attachment_list.append(
                    ue_data_for_current_tick.loc[
                        (ue_data_for_current_tick["ue_id"] == ue_id)
                        & (ue_data_for_current_tick["cell_id"] == past_cell_id)
                    ].iloc[0]
                )

    # Convert the list of attachments to a DataFrame
    current_attachment = pd.DataFrame(current_attachment_list).reset_index(drop=True)

    return current_attachment


def _check_rlf_threshold(df: pd.DataFrame, current_tick_df: pd.DataFrame, rlf_threshold: float) -> pd.DataFrame:
    """
    Updates the dataframe based on the SINR threshold and data from `current_tick_df`.

    For each `ue_id`, if SINR in the dataframe is below `rlf_threshold`:
    - Attempts to update using the best SINR from `current_tick_df`.
    - If no update is found, marks the row as RLF by setting key fields to fallback values.

    Parameters:
        df (pd.DataFrame): Current UE data.
        current_tick_df (pd.DataFrame): Most recent tick-level UE data.
        rlf_threshold (float): SINR threshold for fallback.


      +------------------------------------------------------+
      | df                                                   |
      +--------+---------+------------------+------+---------+
      | ue_id  | cell_id | cell_rxpower_dbm | tick | sinr_db |
      +========+=========+===================================+
      |   0    |    1    |   -100.311970    |  1   | 10.517  |
      |   0    |    2    |    -99.841523    |  1   | 12.125  |
      |   1    |    1    |   -100.294405    |  1   | 11.229  |
      |   1    |    2    |   -100.132420    |  1   | 11.672  |
      |   0    |    1    |   -100.311970    |  2   | 13.930  |
      |   0    |    2    |    -99.841523    |  2   | 14.739  |
      |   1    |    1    |   -100.294405    |  2   | 12.229  |
      |   1    |    2    |   -100.132420    |  2   | 15.222  |
      +--------+---------+------------------+------+---------+

      +------------------------------------------------------+
      | current_tick_df                                      |
      +--------+---------+------------------+------+---------+
      | ue_id  | cell_id | cell_rxpower_dbm | tick | sinr_db |
      +========+=========+===================================+
      |   0    |    1    |   -100.311970    |  1   | 10.517  |
      |   0    |    2    |    -99.841523    |  1   | 12.125  |
      |   1    |    1    |   -100.294405    |  1   | 11.229  |
      |   1    |    2    |   -100.132420    |  1   | 11.672  |
      |   0    |    1    |   -100.311970    |  2   | 13.930  |
      |   0    |    2    |    -99.841523    |  2   | 14.739  |
      |   1    |    1    |   -100.294405    |  2   | 12.229  |
      |   1    |    2    |   -100.132420    |  2   | 15.222  |
      +--------+---------+------------------+------+---------+

    Returns:
        pd.DataFrame: Updated DataFrame with applied fallback logic.
    """
    # Create a copy to avoid modifying the original
    updated_df = df.copy()

    # Find the best SINR row for each ue_id in current_tick_df (vectorized)
    best_rows = current_tick_df.loc[current_tick_df.groupby("ue_id")["sinr_db"].idxmax()]
    best_rows = best_rows[best_rows["sinr_db"] >= rlf_threshold].set_index("ue_id")

    # Vectorized mask for rows below threshold
    below_thresh_mask = updated_df["sinr_db"] < rlf_threshold

    # Try to update with best row from current_tick_df if available and above threshold
    for idx in updated_df[below_thresh_mask].index:
        ue_id = updated_df.at[idx, "ue_id"]
        if ue_id in best_rows.index:
            for col in updated_df.columns.intersection(current_tick_df.columns):
                if col == "ue_id":
                    continue  # Do not update 'ue_id' column
                updated_df.at[idx, col] = best_rows.at[ue_id, col]
        else:
            # If no valid cell, set RLF
            updated_df.at[idx, "sinr_db"] = -np.inf
            updated_df.at[idx, "cell_id"] = "RLF"
            updated_df.at[idx, "cell_rxpower_dbm"] = -np.inf

    return updated_df


def _check_hyst_in_current_tick(
    ue_data_for_current_tick: pd.DataFrame,
    current_attachment: pd.DataFrame,
    past_attachment: pd.DataFrame,
    hyst: float,
) -> pd.DataFrame:
    """
    Applies a hyst check to the data in the current timestamp to prevent unnecessary handovers.
    If the new cell's signal is not stronger than the previous cell by at least `hyst` dB,
    the UE remains attached to the same cell as the previous tick.

    Parameters:
         ue_data_for_current_tick (pd.DataFrame): Signal strength data for UE–cell pairs in the current tick.
        current_attachment (pd.DataFrame): Proposed attachment decisions for the current tick.
        past_attachment (pd.DataFrame): Previous tick’s attachment state.
        hyst (float): Hysteresis threshold in dB.

    +---------+--------+------------------+
    | cell_id | ue_id  | cell_rxpower_dbm |
    +=========+========+==================+
    |    1    |   0    |   -100.311970    |
    |    2    |   0    |    -99.841523    |
    |    1    |   1    |   -100.294405    |
    |    2    |   1    |   -100.132420    |
    |    1    |   2    |   -100.650003    |
    |    2    |   2    |   -100.456381    |
    |    1    |   3    |   -100.987321    |
    |    2    |   3    |   -100.864529    |
    +---------+--------+------------------+

    Returns:
        pd.DataFrame: Updated UE–cell attachments after hysteresis filtering.
    """
    if current_attachment.shape != past_attachment.shape:
        raise AssertionError(
            "current attachment and past attachment are not consistent. Check their shape, ue_id and cell_id columns."
        )
    elif set(current_attachment["ue_id"]) != set(past_attachment["ue_id"]):
        raise AssertionError("current attachment and past attachment have different ue_ids.")
    for i, curr in current_attachment.iterrows():
        prev = past_attachment[past_attachment["ue_id"] == curr["ue_id"]].iloc[0]
        # * ignoring if no cell switch for this UE
        if curr["cell_id"] == prev["cell_id"]:
            continue
        # * get rxpowers for current and previous attached cell calculated in current tick
        curr_attachment_rxpower = ue_data_for_current_tick[
            (ue_data_for_current_tick["ue_id"] == curr["ue_id"])
            & (ue_data_for_current_tick["cell_id"] == curr["cell_id"])
        ]["cell_rxpower_dbm"].values[0]
        try:
            prev_attachment_rxpower = ue_data_for_current_tick[
                (ue_data_for_current_tick["ue_id"] == prev["ue_id"])
                & (ue_data_for_current_tick["cell_id"] == prev["cell_id"])
            ]["cell_rxpower_dbm"].values[0]
        except IndexError:  # If no previous attachment found in current tick, set to -inf
            prev_attachment_rxpower = -np.inf
        # * hysteresis check & revert if failed
        if curr_attachment_rxpower < prev_attachment_rxpower + hyst:
            current_attachment.at[i, "cell_id"] = prev["cell_id"]
            current_attachment.at[i, "cell_rxpower_dbm"] = prev_attachment_rxpower

    return current_attachment
