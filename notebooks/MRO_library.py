import pandas as pd
import numpy as np
import math
import apps.coverage_capacity_optimization.constants as constants


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
        serving_row = group.loc[group['cell_rxpower_dbm'].idxmax()]
        serving_rsrp_dbm = serving_row['cell_rxpower_dbm']

        # Convert background noise from dB to linear scale.
        noise_linear = 10 ** (constants.LATENT_BACKGROUND_NOISE_DB / 10)

        # Convert all received powers from dBm to linear scale and sum them.
        total_power_linear = np.sum(10 ** (group['cell_rxpower_dbm'] / 10))

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

    # Group by mock_ue_id. Assume each UE might have multiple measurements (e.g., different cells).
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