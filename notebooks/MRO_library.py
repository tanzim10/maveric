import pandas as pd
import numpy as np
import math
import apps.coverage_capacity_optimization.constants as constants


def count_switches(df):
    """
    Counts the number of times a mock_ue_id switches cell_id to a different one (excluding 'RLF').
    """
    count = 0
    df = df.sort_values(by=['ue_id', 'tick'])  # Ensure correct order
    prev_cells = {}
    prev_ticks = {}

    for _, row in df.iterrows():
        ue_id, cell_id, tick = row['ue_id'], row['cell_id'], row['tick']

        if ue_id in prev_cells and prev_cells[ue_id] != cell_id and prev_cells[ue_id] is not None:
            if tick == prev_ticks[ue_id] + 1 and cell_id != "RLF":
                count += 1

        prev_cells[ue_id] = cell_id
        prev_ticks[ue_id] = tick
    return count

def count_rlf(df):
    """
    Counts the number of times a mock_ue_id switches from any cell_id to 'RLF'.
    """
    return (df['cell_id'] == "RLF").sum()

def calculate_mro_metric(data):
    # Constants for interruption times
    ts = 50 / 1000  # Convert ms to seconds
    t_nas = 1000 / 1000  # Convert ms to seconds

    # Calculate total time (T) based on ticks; assuming each tick represents a uniform time slice
    # This could be adjusted if ticks represent variable time slices
    # Rather than passing the UE Data as whole we can send just an integar for tick
    ticks = len(data['tick'].unique())
    # Assuming each tick represents 50ms (this value may need to be adjusted based on actual data characteristics)
    tick_duration_seconds = 1  # 1 second per tick
    T = ticks * tick_duration_seconds

    ns_handover_count = count_switches(data)  # Count of handovers to different cells (excluding RLF)
    nf_handover_count = count_rlf(data)  # Count of handovers to RLF

    # Calculate D
    D = T - (ns_handover_count * ts + nf_handover_count * t_nas)

    return D

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Earth radius in kilometers
    return constants.RADIUS_EARTH_EQUATOR_KM * c


def calculate_received_power(distance_km, frequency_mhz):
    """
    Calculate received power using the Free-Space Path Loss (FSPL) model.

    Parameters:
    - tx_power_dbm: Transmit power in dBm.
    - distance_km: Distance between UE and cell in kilometers.
    - frequency_mhz: Frequency in MHz.

    Returns:
    - Received power in dBm.
    """
    tx_power_dbm = constants.tx_power_dbm

    # Convert distance to meters
    distance_m = distance_km * 1000

    # Free-Space Path Loss formula
    fspl_db = (
        20 * np.log10(distance_m) + 20 * np.log10(frequency_mhz) - 27.55
    )  # FSPL in dB

    # Received power
    received_power_dbm = tx_power_dbm - fspl_db
    return received_power_dbm


def concatenate_ue_to_topology(ue_data, topology):
    # Initialize an empty list to store the results
    results = []

    # Loop through each UE and calculate the distance to each cell
    for _, ue_row in ue_data.iterrows():
        ue_lat = ue_row["latitude"]
        ue_lon = ue_row["longitude"]

        # Calculate the distance to each cell in topology_1
        distances = topology.apply(
            lambda row: haversine(ue_lat, ue_lon, row["cell_lat"], row["cell_lon"]),
            axis=1,
        )

        # Find the index of the closest cell
        closest_cell_idx = distances.idxmin()

        # Get the data of the closest cell
        closest_cell = topology.iloc[closest_cell_idx]

        # Create a new row with all data from ue_data and topology_1 (closest cell)
        combined_data = {
            "mock_ue_id": ue_row["mock_ue_id"],
            "longitude": ue_row["longitude"],
            "latitude": ue_row["latitude"],
            "tick": ue_row["tick"],
            "cell_lat": closest_cell["cell_lat"],
            "cell_lon": closest_cell["cell_lon"],
            "cell_id": closest_cell["cell_id"],
            "cell_az_deg": closest_cell["cell_az_deg"],
            "cell_carrier_freq_mhz": closest_cell["cell_carrier_freq_mhz"],
        }

        # Append the combined data to the results list
        results.append(combined_data)

    # Convert the results into a DataFrame
    full_data = pd.DataFrame(results)
    return full_data


def connect_ue_to_all_cells(ue_data, topology):
    # Initialize an empty list to store the results
    results = []

    # Loop through each UE and associate it with every cell
    for _, ue_row in ue_data.iterrows():
        ue_lat = ue_row["latitude"]
        ue_lon = ue_row["longitude"]

        # Loop through each cell in topology
        for _, cell_row in topology.iterrows():
            # Calculate the distance to the current cell
            distance = haversine(
                ue_lat, ue_lon, cell_row["cell_lat"], cell_row["cell_lon"]
            )

            # Create a new row combining UE data with each cell's data
            combined_data = {
                "mock_ue_id": ue_row["mock_ue_id"],
                "longitude": ue_row["longitude"],
                "latitude": ue_row["latitude"],
                "tick": ue_row["tick"],
                "cell_lat": cell_row["cell_lat"],
                "cell_lon": cell_row["cell_lon"],
                "cell_id": cell_row["cell_id"],
                "cell_az_deg": cell_row["cell_az_deg"],
                "cell_carrier_freq_mhz": cell_row["cell_carrier_freq_mhz"],
                "distance": distance,  # Optionally include the distance if needed
            }

            # Append the combined data to the results list
            results.append(combined_data)

    # Convert the results into a DataFrame
    full_data = pd.DataFrame(results)
    full_data.drop(columns=["distance"], inplace=True)
    return full_data


def add_sinr_column(df: pd.DataFrame) -> pd.DataFrame:
    # Function to compute SINR for a group of rows corresponding to one frequency layer.
    def compute_layer_sinr(group: pd.DataFrame) -> float:
        # Get the RSRP: the highest received power in this frequency group.
        # Using idxmax ensures we pick the row with max cell_rxpower_dbm.
        serving_row = group.loc[group["cell_rxpower_dbm"].idxmax()]
        serving_rsrp_dbm = serving_row["cell_rxpower_dbm"]

        # Convert background noise from dB to linear scale.
        noise_linear = 10 ** (constants.LATENT_BACKGROUND_NOISE_DB / 10)

        # Convert all received powers from dBm to linear scale and sum them.
        total_power_linear = np.sum(10 ** (group["cell_rxpower_dbm"] / 10))

        # Interference is total power minus the serving cell power (in linear scale)
        serving_power_linear = 10 ** (serving_rsrp_dbm / 10)
        interference_linear = total_power_linear - serving_power_linear

        # Total interference plus noise in dBm
        total_interference_noise_dbm = 10 * np.log10(interference_linear + noise_linear)

        # SINR is the difference between serving RSRP and the total interference-plus-noise (both in dBm)
        sinr_db = serving_rsrp_dbm - total_interference_noise_dbm

        return sinr_db

    # This dictionary will store the computed SINR per UE id.
    ue_sinr = {}

    # Group by ue_id. Assume each UE might have multiple measurements (e.g., different cells).
    for ue_id, ue_group in df.groupby("ue_id"):
        # Group by frequency as interference is only calculated among cells on the same frequency.
        sinr_by_freq = {}
        for freq, freq_group in ue_group.groupby("cell_carrier_freq_mhz"):
            sinr_by_freq[freq] = compute_layer_sinr(freq_group)

        # Select the frequency layer with the highest SINR as the final value for the UE.
        # If there is only one frequency group, this simply takes that value.
        max_sinr = max(sinr_by_freq.values())
        ue_sinr[ue_id] = max_sinr

    # Map the computed SINR back to the original dataframe rows.
    df = df.copy()
    df["sinr_db"] = df["ue_id"].map(ue_sinr)
    return df


def _check_hyst(ue_data_for_current_tick, past_attachment, hyst):
    """
    Function to update UE data based on hysteresis condition and past attachment.
    It selects the best data between current tick and past attachment for each ue_id.
    """
    # Merge the current tick data with past attachment data (to compare past power)
    merged_df = pd.merge(ue_data_for_current_tick, past_attachment,
                         on='ue_id', how='left', suffixes=('', '_past'))

    # Initialize an empty list to store the final rows
    final_data = []

    # Group by 'ue_id' to process each UE individually
    for ue_id, group in merged_df.groupby('ue_id'):
        # Initialize variables to track the best row for the ue_id
        best_row = None
        best_power = -999  # Start with an arbitrarily low value for comparison (can also use NaN)

        # Iterate through each row (cell_id) for the current ue_id
        for _, row in group.iterrows():
            current_power = row['cell_rxpower_dbm']
            past_cell_id = row['cell_id_past']  # Cell ID from past attachment

            # Retrieve the past data for the previous cell_id of this ue_id
            past_data = ue_data_for_current_tick[(ue_data_for_current_tick['ue_id'] == ue_id) &
                                                 (ue_data_for_current_tick['cell_id'] == past_cell_id)]

            # If past data exists, get the past power
            if not past_data.empty:
                past_power = past_data.iloc[0]['cell_rxpower_dbm']
            else:
                past_power = -999  # Default value if no past data found

            # Check if the current power exceeds the past power by at least 'hyst'
            if current_power - past_power >= hyst:
                # If the condition is met, consider the current row (i.e., current power is better)
                if current_power > best_power:  # Keep the row with the highest power
                    best_row = row
                    best_power = current_power
            else:
                # Otherwise, consider the past data (if it has higher power)
                if past_power > best_power:
                    best_row = past_data.iloc[0]  # Select the past data row
                    best_power = past_power

        # After processing all cell_ids for this ue_id, add the best row to the final list
        if best_row is not None:
            final_data.append(best_row)

    # Convert the final list of rows into a DataFrame
    final_df = pd.DataFrame(final_data)

    # Remove columns that have the '_past' suffix (since we only need current data)
    final_df = final_df.loc[:, ~final_df.columns.str.endswith('_past')]

    return final_df


def _check_ttt(strongest_server_history, ue_data_for_current_tick, past_attachment):
    current_attachment_list = []  # contains updated ue -> cell + current data
    merged_df = pd.concat(strongest_server_history, ignore_index=True)

    # individual UE scope
    for UE in ue_data_for_current_tick["ue_id"].unique():
        # Check consistency for this specific UE
        history_consistency_check = merged_df[merged_df["ue_id"] == UE][
            "cell_id"
        ].nunique()
        consistent_cell_id = merged_df[merged_df["ue_id"] == UE]["cell_id"].unique()[0]

        # attach consistent cell or past cell decision
        if history_consistency_check == 1:
            current_attachment_list.append(
                ue_data_for_current_tick[
                    (ue_data_for_current_tick["ue_id"] == UE)
                    & (ue_data_for_current_tick["cell_id"] == consistent_cell_id)
                ].iloc[0]
            )
        else:
            past_cell_id = past_attachment[past_attachment["ue_id"] == UE][
                "cell_id"
            ].values[0]
            # logger.info("%s", past_cell_id)
            if past_cell_id == "RLF":
                # Select the cell with the highest cell_rxpower_dbm for this UE
                highest_power_row = (
                    ue_data_for_current_tick[ue_data_for_current_tick["ue_id"] == UE]
                    .nlargest(1, "cell_rxpower_dbm")
                    .iloc[0]
                )
                current_attachment_list.append(highest_power_row)
            else:
                current_attachment_list.append(
                    ue_data_for_current_tick[
                        (ue_data_for_current_tick["ue_id"] == UE)
                        & (ue_data_for_current_tick["cell_id"] == past_cell_id)
                    ].iloc[0]
                )

    current_attachment = pd.DataFrame(current_attachment_list).reset_index(drop=True)

    return current_attachment


def _check_rlf_threshold(df, current_tick_df, rlf_threshold):
    # Create a copy of the original df to avoid modifying it in place
    updated_df = df.copy()

    # Iterate over the unique ue_ids in the original df
    for ue_id in df["ue_id"]:
        # Get the row for the current ue_id in both dataframes
        df_ue = df[df["ue_id"] == ue_id]
        current_tick_ue = current_tick_df[current_tick_df["ue_id"] == ue_id]

        # Extract the SINR value from the current row in df
        max_sinr = df_ue["sinr_db"].values[0]

        if max_sinr >= rlf_threshold:
            # If the max SINR is greater than or equal to the threshold, no change is needed
            continue
        else:
            # If the SINR in df is below the threshold, check current_tick_df for the best SINR
            max_sinr_current_tick = current_tick_ue["sinr_db"].max()

            if max_sinr_current_tick >= rlf_threshold:
                # If the best SINR in current_tick_df is above the threshold, update the data
                updated_df.loc[
                    updated_df["ue_id"] == ue_id, "sinr_db"
                ] = max_sinr_current_tick
                updated_df.loc[
                    updated_df["ue_id"] == ue_id, "cell_id"
                ] = current_tick_ue["cell_id"].values[0]
                updated_df.loc[
                    updated_df["ue_id"] == ue_id, "cell_rxpower_dbm"
                ] = current_tick_ue["cell_rxpower_dbm"].values[0]
                updated_df.loc[
                    updated_df["ue_id"] == ue_id, "cell_lat"
                ] = current_tick_ue["cell_lat"].values[0]
                updated_df.loc[
                    updated_df["ue_id"] == ue_id, "cell_lon"
                ] = current_tick_ue["cell_lon"].values[0]
                updated_df.loc[
                    updated_df["ue_id"] == ue_id, "cell_carrier_freq_mhz"
                ] = current_tick_ue["cell_carrier_freq_mhz"].values[0]
                updated_df.loc[
                    updated_df["ue_id"] == ue_id, "cell_az_deg"
                ] = current_tick_ue["cell_az_deg"].values[0]
                updated_df.loc[
                    updated_df["ue_id"] == ue_id, "distance_km"
                ] = current_tick_ue["distance_km"].values[0]
                updated_df.loc[
                    updated_df["ue_id"] == ue_id, "relative_bearing"
                ] = current_tick_ue["relative_bearing"].values[0]

            else:
                # If both are below the threshold, set RLF values
                updated_df.loc[updated_df["ue_id"] == ue_id, "sinr_db"] = -np.inf
                updated_df.loc[updated_df["ue_id"] == ue_id, "cell_id"] = "RLF"
                updated_df.loc[
                    updated_df["ue_id"] == ue_id, "cell_rxpower_dbm"
                ] = -np.inf

    return updated_df


def _check_hyst_in_current_tick(
    ue_data_for_current_tick: pd.DataFrame,
    current_attachment: pd.DataFrame,
    past_attachment: pd.DataFrame,
    hyst: float,
) -> pd.DataFrame:
    if current_attachment.shape != past_attachment.shape:
        raise AssertionError(
            "current attachment and past attachment are not consistent. Check their shape, ue_id and cell_id columns."
        )
    elif set(current_attachment["ue_id"]) != set(past_attachment["ue_id"]):
        raise AssertionError(
            "current attachment and past attachment have different ue_ids."
        )
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
        except:
            prev_attachment_rxpower = -np.inf
        # * hysteresis check & revert if failed
        if curr_attachment_rxpower < prev_attachment_rxpower + hyst:
            current_attachment.at[i, "cell_id"] = prev["cell_id"]
            current_attachment.at[i, "cell_rxpower_dbm"] = prev_attachment_rxpower

    return current_attachment


def _perform_attachment_hyst_ttt_per_tick(
    ue_data_for_current_tick,
    strongest_server_history,
    past_attachment,
    ttt,
    hyst,
    use_strongest_server=False,
):
    current_strongest = ue_data_for_current_tick.loc[
        ue_data_for_current_tick.groupby("ue_id")["cell_rxpower_dbm"].idxmax()
    ]
    if len(strongest_server_history) >= ttt:
        raise AssertionError(
            "Error: Strongest_Server_History needs to be Less Than TTT!"
        )
    else:
        if use_strongest_server:
            current_attachment = current_strongest
            strongest_server_history.append(current_strongest)

        else:
            if ttt == len(strongest_server_history) + 1:
                current_strongest = _check_hyst(
                    ue_data_for_current_tick, past_attachment, hyst
                )
                strongest_server_history.append(current_strongest)
                current_attachment = _check_ttt(
                    strongest_server_history, ue_data_for_current_tick, past_attachment
                )
                current_attachment = _check_hyst_in_current_tick(
                    ue_data_for_current_tick, current_attachment, past_attachment, hyst
                )
            else:
                raise AssertionError(
                    "Length of Strongest Server History must be EQUALS to TTT - 1."
                    "Call Perform Attachment with use_strongest_server = True"
                )

    if len(strongest_server_history) == ttt:
        strongest_server_history.pop(0)

    return strongest_server_history, current_attachment


def perform_attachment_hyst_ttt(ue_data, hyst, ttt, rlf_threshold):
    strongest_server_history = []
    current_attachment = pd.DataFrame()

    tick_dataframes = {}
    # Group the data by tick
    for tick in sorted(ue_data["tick"].unique()):
        tick_dataframes[tick] = ue_data[ue_data["tick"] == tick].copy()

    cell_attached_df = pd.DataFrame()
    for tick in range(len(tick_dataframes)):
        if ttt - 1 > len(strongest_server_history):
            (
                strongest_server_history,
                current_attachment,
            ) = _perform_attachment_hyst_ttt_per_tick(
                tick_dataframes[tick],
                strongest_server_history,
                current_attachment,
                ttt,
                hyst,
                use_strongest_server=True,
            )
        else:
            (
                strongest_server_history,
                current_attachment,
            ) = _perform_attachment_hyst_ttt_per_tick(
                tick_dataframes[tick],
                strongest_server_history,
                current_attachment,
                ttt,
                hyst,
                use_strongest_server=False,
            )
        current_attachment = _check_rlf_threshold(
            current_attachment, tick_dataframes[tick], rlf_threshold
        )
        cell_attached_df = pd.concat([cell_attached_df, current_attachment])

    return cell_attached_df
