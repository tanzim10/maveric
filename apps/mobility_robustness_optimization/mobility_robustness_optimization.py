import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from notebooks.radp_library import calc_log_distance, calc_relative_bearing, calculate_received_power, get_percell_data
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, NormMethod
from radp.digital_twin.utils import constants
from radp.digital_twin.utils.cell_selection import perform_attachment
from radp.digital_twin.utils.gis_tools import GISTools


class MobilityRobustnessOptimization(ABC):
    """
    A class that contains a prototypical proof-of-concept of an `Mobility Robustness Optimization (MRO)` RIC xApp.
    """

    def __init__(
        self,
        mobility_model_params: Dict[str, Dict],
        topology: pd.DataFrame,
        bdt: Optional[Dict[str, BayesianDigitalTwin]] = None,
    ):
        self.topology = topology
        self.bayesian_digital_twins = bdt if bdt is not None else {}
        self.mobility_model_params = mobility_model_params
        self.training_data = None
        self.prediction_data = None
        self.update_data = None
        self.simulation_data = None

    def train_or_update_rf_twin(self, new_data: pd.DataFrame):
        try:
            if not isinstance(new_data, pd.DataFrame):
                raise TypeError("The input 'new_data' must be a pandas DataFrame.")

            # TODO: write a function to check if the data is in cartesian format or not

            expected_columns = {"longitude", "latitude", "cell_id", "cell_rxpwr_dbm"}
            if not expected_columns.issubset(new_data.columns):
                raise ValueError(f"The input DataFrame must contain the following columns: {expected_columns}")

            # TODO: add comment to explain the purpose of converting the cell_id to string
            new_data["cell_id"] = new_data["cell_id"].apply(lambda x: f"cell_{x}")
            self.topology["cell_id"] = self.topology["cell_id"].apply(lambda x: f"cell_{x}")

            prepared_data = self._prepare_train_or_update_data(new_data)

            if self.bayesian_digital_twins:
                print("Updating existing Bayesian Digital Twins with new data.")
                # TODO: Add Update Logic Here
                # self.update_data = new_data
                # updated_data = self._preprocess_ue_update_data()

                # prepared_data_list = list(prepared_data.values())
                print(self.bayesian_digital_twins)
                print(self.topology)
                # for data_idx, update_data_df in enumerate(prepared_data_list):
                #     update_cell_id = data_idx + 1
                #     if update_cell_id in self.bayesian_digital_twins:
                #         self.bayesian_digital_twins[
                #             update_cell_id
                #         ].update_trained_gpmodel([update_data_df])
            else:
                print("No Bayesian Digital Twins available for update. Training from scratch.")
                self._training(maxiter=100, train_data=prepared_data)

                # TODO: Add Training From Scratch Logic
                # self._training(maxiter=100, train_data=new_data)
        except TypeError as te:
            print(f"TypeError: {te}")
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except KeyError as ke:
            print(f"KeyError: {ke}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def save(bayesian_digital_twins, file_loc):
        """
        Saves the Bayesian Digital Twins to a pickle file. Returns `True` if saving succeeds,
        and `NotImplemented` if it fails.

        """
        filename = f"{file_loc}/digital_twins.pkl"
        try:
            if not isinstance(bayesian_digital_twins, dict):
                raise TypeError("The input 'bayesian_digital_twins' must be a dictionary.")

            # Ensure the directory exists
            os.makedirs(file_loc, exist_ok=True)

            with open(filename, "wb") as fp:
                pickle.dump(bayesian_digital_twins, fp)

            print("Twins Saved Successfully as Pickle.")
            return True  # Indicate successful save
        except TypeError as te:
            print(f"TypeError: {te}")
        except OSError as oe:
            print(f"OSError: {oe}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    @abstractmethod
    def solve(self):
        """
        Solve the mobility robustness optimization problem.

        This method is an abstract method that must be implemented by its subclasses.
        """
        pass

    def _training(self, maxiter: int, train_data: pd.DataFrame) -> List[float]:
        """
        Trains the Bayesian Digital Twins for each cell in the topology using the UE locations and features
        like log distance, relative bearing, and cell received power (Rx power).
        """
        training_data = train_data
        bayesian_digital_twins = {}
        loss_vs_iters = []
        for train_cell_id, training_data_idx in training_data.items():
            bayesian_digital_twins[train_cell_id] = BayesianDigitalTwin(
                data_in=[training_data_idx],
                x_columns=["log_distance", "relative_bearing"],
                y_columns=["cell_rxpwr_dbm"],
                norm_method=NormMethod.MINMAX,
            )
            self.bayesian_digital_twins[train_cell_id] = bayesian_digital_twins[train_cell_id]
            loss_vs_iters.append(
                bayesian_digital_twins[train_cell_id].train_distributed_gpmodel(
                    maxiter=maxiter,
                )
            )
        return loss_vs_iters

    def _prepare_train_or_update_data(self, df):
        required_columns = {"cell_lat", "cell_lon", "cell_az_deg"}
        if not required_columns.issubset(df.columns):
            df = self.add_cell_info(df, self.topology)

        self.update_data = calc_log_distance(df)
        self.update_data = calc_relative_bearing(self.update_data)

        self.update_data = self.update_data.loc[:, ["cell_id", "log_distance", "relative_bearing", "cell_rxpwr_dbm"]]

        train_per_cell_df = [x for _, x in self.update_data.groupby("cell_id")]
        n_cell = len(self.topology.index)

        metadata_df = pd.DataFrame(
            {
                "cell_id": [cell_id for cell_id in self.topology.cell_id],
                "idx": [i + 1 for i in range(n_cell)],
            }
        )

        idx_cell_id_mapping = dict(zip(metadata_df.idx, metadata_df.cell_id))
        n_samples_train = []

        for df in train_per_cell_df:
            n_samples_train.append(df.shape[0])

        train_per_cell_df_processed = []
        for i in range(n_cell):
            train_per_cell_df_processed.append(
                get_percell_data(
                    data_in=train_per_cell_df[i],
                    choose_strongest_samples_percell=False,
                    n_samples=n_samples_train[i],
                )[0][0]
            )

        training_data = {}

        for i, df in enumerate(train_per_cell_df_processed):
            train_cell_id = idx_cell_id_mapping[i + 1]
            training_data[train_cell_id] = df

        return training_data

    def add_cell_info(self, new_data_with_rx_data, topology):
        """
        Adds cell information ['cell_id', 'cell_lat', 'cell_lon', 'cell_az_deg']
        to the DataFrame based on cell_id.
        Converts integer cell_id to string format like 'cell_1' to match topology.
        """
        # Convert int to str format matching topology: 'cell_1', 'cell_2', etc.
        if new_data_with_rx_data["cell_id"].dtype == int:
            new_data_with_rx_data["cell_id"] = new_data_with_rx_data["cell_id"].apply(lambda x: f"cell_{x}")

        # Merge using consistent cell_id format
        new_data_topology_merged = new_data_with_rx_data.merge(
            topology[["cell_id", "cell_lat", "cell_lon", "cell_az_deg"]], on="cell_id", how="left"
        )

        return new_data_topology_merged

    def _predictions(self, pred_data) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predicts the received power for each User Equipment (UE) at different locations
        and ticks using Bayesian Digital Twins.
        It then determines the best cell for each UE to attach based on the predicted power values.
        """
        self.prediction_data = pred_data
        prediction_data = self._preprocess_prediction_data()
        full_prediction_df = pd.DataFrame()

        # Loop over each 'tick'
        for tick, tick_df in prediction_data.groupby("tick"):
            # Loop over each 'cell_id' within the current 'tick'
            for cell_id, cell_df in tick_df.groupby("cell_id"):
                cell_id = f"cell_{cell_id}"  # FIXME: should look better
                # Check if the Bayesian model for this cell_id exists
                if cell_id in self.bayesian_digital_twins:
                    # Perform the Bayesian prediction
                    pred_means_percell, _ = self.bayesian_digital_twins[cell_id].predict_distributed_gpmodel(
                        prediction_dfs=[cell_df]
                    )

                    # Assuming 'pred_means_percell' returns a list of predictions corresponding to the DataFrame index
                    cell_df["pred_means"] = pred_means_percell[0]

                    # Include additional necessary columns for the final DataFrame
                    cell_df["tick"] = tick
                    cell_df["cell_id"] = cell_id

                    # Append the predictions to the full DataFrame
                    full_prediction_df = pd.concat([full_prediction_df, cell_df], ignore_index=True)
                else:
                    # Handle missing models, e.g., log a warning or initialize a default model
                    print(f"No model available for cell_id {cell_id}, skipping prediction.")

        full_prediction_df = full_prediction_df.rename(columns={"latitude": "loc_y", "longitude": "loc_x"})
        predicted = perform_attachment(full_prediction_df, self.topology)

        return predicted, full_prediction_df

    def _prepare_all_UEs_from_all_cells_df(
        self, prediction: bool = False, simulation: bool = False, update: bool = False
    ) -> pd.DataFrame:
        """
        Connects each user equipment (UE) entry to all cells in the topology for each tick,
        effectively creating a Cartesian product of UEs and cells, which includes data from both sources.
        """

        if prediction:
            ue_data_tmp = self.prediction_data.copy()
        elif simulation:
            ue_data_tmp = self.simulation_data.copy()
        elif update:
            ue_data_tmp = self.update_data.copy()
        else:
            ue_data_tmp = self.training_data.copy()
        topology_tmp = self.topology.copy()
        # Remove the 'cell_' prefix and convert cell_id to integer if needed
        if topology_tmp["cell_id"].dtype == object:
            topology_tmp["cell_id"] = topology_tmp["cell_id"].str.replace("cell_", "").astype(int)
        ue_data_tmp["key"] = 1
        topology_tmp["key"] = 1
        combined_df = pd.merge(ue_data_tmp, topology_tmp, on="key").drop("key", axis=1)
        return combined_df

    def _preprocess_ue_topology_data(self) -> pd.DataFrame:
        full_data = self._prepare_all_UEs_from_all_cells_df()
        full_data["log_distance"] = full_data.apply(
            lambda row: GISTools.get_log_distance(row["latitude"], row["longitude"], row["cell_lat"], row["cell_lon"]),
            axis=1,
        )

        full_data["cell_rxpwr_dbm"] = full_data.apply(
            lambda row: calculate_received_power(row["log_distance"], row["cell_carrier_freq_mhz"]),
            axis=1,
        )

        return full_data

    def _preprocess_ue_training_data(self) -> pd.DataFrame:
        data = self._preprocess_ue_topology_data()
        train_per_cell_df = [x for _, x in data.groupby("cell_id")]
        n_cell = len(self.topology.index)

        metadata_df = pd.DataFrame(
            {
                "cell_id": [cell_id for cell_id in self.topology.cell_id],
                "idx": [i + 1 for i in range(n_cell)],
            }
        )
        idx_cell_id_mapping = dict(zip(metadata_df.idx, metadata_df.cell_id))
        # desired_idxs = [1 + r for r in range(n_cell)]

        n_samples_train = []
        for df in train_per_cell_df:
            n_samples_train.append(df.shape[0])

        train_per_cell_df_processed = []
        for i in range(n_cell):
            train_per_cell_df_processed.append(
                get_percell_data(
                    data_in=train_per_cell_df[i],
                    choose_strongest_samples_percell=False,
                    n_samples=n_samples_train[i],
                )[0][0]
            )

        training_data = {}

        for i, df in enumerate(train_per_cell_df_processed):
            train_cell_id = idx_cell_id_mapping[i + 1]
            training_data[train_cell_id] = df

        for train_cell_id, training_data_idx in training_data.items():
            training_data_idx["cell_id"] = train_cell_id
            training_data_idx["cell_lat"] = self.topology[self.topology["cell_id"] == train_cell_id]["cell_lat"].values[
                0
            ]
            training_data_idx["cell_lon"] = self.topology[self.topology["cell_id"] == train_cell_id]["cell_lon"].values[
                0
            ]
            training_data_idx["cell_az_deg"] = self.topology[self.topology["cell_id"] == train_cell_id][
                "cell_az_deg"
            ].values[0]
            training_data_idx["cell_carrier_freq_mhz"] = self.topology[self.topology["cell_id"] == train_cell_id][
                "cell_carrier_freq_mhz"
            ].values[0]
            training_data_idx["relative_bearing"] = [
                GISTools.get_relative_bearing(
                    training_data_idx["cell_az_deg"].values[0],
                    training_data_idx["cell_lat"].values[0],
                    training_data_idx["cell_lon"].values[0],
                    lat,
                    lon,
                )
                for lat, lon in zip(training_data_idx["latitude"], training_data_idx["longitude"])
            ]

        return training_data

    def _preprocess_ue_update_data(self) -> pd.DataFrame:
        data = self._prepare_all_UEs_from_all_cells_df(update=True)
        data["log_distance"] = data.apply(
            lambda row: GISTools.get_log_distance(row["latitude"], row["longitude"], row["cell_lat"], row["cell_lon"]),
            axis=1,
        )

        data["cell_rxpwr_dbm"] = data.apply(
            lambda row: calculate_received_power(row["log_distance"], row["cell_carrier_freq_mhz"]),
            axis=1,
        )

        update_per_cell_df = [x for _, x in data.groupby("cell_id")]
        n_cell = len(self.topology.index)

        metadata_df = pd.DataFrame(
            {
                "cell_id": [cell_id for cell_id in self.topology.cell_id],
                "idx": [i + 1 for i in range(n_cell)],
            }
        )
        idx_cell_id_mapping = dict(zip(metadata_df.idx, metadata_df.cell_id))

        n_samples_update = []
        for df in update_per_cell_df:
            n_samples_update.append(df.shape[0])

        update_per_cell_df_processed = []
        for i in range(n_cell):
            update_per_cell_df_processed.append(
                get_percell_data(
                    data_in=update_per_cell_df[i],
                    choose_strongest_samples_percell=False,
                    n_samples=n_samples_update[i],
                )[0][0]
            )

        update_data = {}

        for i, df in enumerate(update_per_cell_df_processed):
            update_cell_id = idx_cell_id_mapping[i + 1]
            update_data[update_cell_id] = df

        for update_cell_id, update_data_idx in update_data.items():
            update_data_idx["cell_id"] = update_cell_id
            update_data_idx["cell_lat"] = self.topology[self.topology["cell_id"] == update_cell_id]["cell_lat"].values[
                0
            ]
            update_data_idx["cell_lon"] = self.topology[self.topology["cell_id"] == update_cell_id]["cell_lon"].values[
                0
            ]
            update_data_idx["cell_az_deg"] = self.topology[self.topology["cell_id"] == update_cell_id][
                "cell_az_deg"
            ].values[0]
            update_data_idx["cell_carrier_freq_mhz"] = self.topology[self.topology["cell_id"] == update_cell_id][
                "cell_carrier_freq_mhz"
            ].values[0]
            update_data_idx["relative_bearing"] = [
                GISTools.get_relative_bearing(
                    update_data_idx["cell_az_deg"].values[0],
                    update_data_idx["cell_lat"].values[0],
                    update_data_idx["cell_lon"].values[0],
                    lat,
                    lon,
                )
                for lat, lon in zip(update_data_idx["latitude"], update_data_idx["longitude"])
            ]
        return update_data

    def _preprocess_prediction_data(self) -> pd.DataFrame:
        data = self._prepare_all_UEs_from_all_cells_df(prediction=True)

        data["log_distance"] = data.apply(
            lambda row: GISTools.get_log_distance(row["latitude"], row["longitude"], row["cell_lat"], row["cell_lon"]),
            axis=1,
        )
        data["cell_rxpwr_dbm"] = data.apply(
            lambda row: calculate_received_power(row["log_distance"], row["cell_carrier_freq_mhz"]),
            axis=1,
        )

        data["relative_bearing"] = data.apply(
            lambda row: GISTools.get_relative_bearing(
                row["cell_az_deg"],
                row["cell_lat"],
                row["cell_lon"],
                row["latitude"],
                row["longitude"],
            ),
            axis=1,
        )
        return data

    def _preprocess_simulation_data(self, df) -> pd.DataFrame:
        df.drop(
            columns=["rxpower_stddev_dbm", "rxpower_dbm", "cell_rxpwr_dbm"],
            inplace=True,
        )
        df.rename(
            columns={
                "mock_ue_id": "ue_id",
                "log_distance": "distance_km",
                "pred_means": "cell_rxpower_dbm",
            },
            inplace=True,
        )
        self.topology["cell_id"] = self.topology["cell_id"].str.replace("cell_", "").astype(int)
        df["cell_id"] = df["cell_id"].str.extract(r"(\d+)").astype(int)
        df = self._add_sinr_column(df)
        return df

    def _add_sinr_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'sinr_db' column to the input DataFrame, computing the Signal-to-Interference-plus-Noise Ratio (SINR)
        for each UE–cell pair based on received signal power, background noise, and interference.

        Parameters:
            df (pd.DataFrame): DataFrame with 'ue_id', 'cell_rxpower_dbm', and 'cell_carrier_freq_mhz' per row.

        Returns:
            pd.DataFrame: Updated DataFrame with an additional 'sinr_db' column.
        """
        # Convert background noise from dB to linear scale
        noise_linear = 10 ** (constants.LATENT_BACKGROUND_NOISE_DB / 10)

        # Compute SINR for each row (UE–cell pair), given its group
        def compute_row_level_sinr(row: pd.Series, group: pd.DataFrame) -> float:  # where cell column? [DONE]
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
            signal_dbm = row["cell_rxpower_dbm"]

            # Exclude the current row (serving cell) to compute interference
            interference_linear = np.sum(10 ** (group.loc[group.index != row.name, "cell_rxpower_dbm"] / 10))
            total_interference_plus_noise_linear = interference_linear + noise_linear

            total_interference_plus_noise_dbm = 10 * np.log10(total_interference_plus_noise_linear)
            sinr_db = signal_dbm - total_interference_plus_noise_dbm
            return sinr_db

        # Apply per UE and frequency
        df = df.copy()
        df["sinr_db"] = (
            df.groupby(["ue_id", "cell_carrier_freq_mhz"])
            .apply(lambda group: group.apply(lambda row: compute_row_level_sinr(row, group), axis=1))
            .reset_index(level=[0, 1], drop=True)
        )

        return df


# Functions for MRO metrics and Handover events
def _count_handovers(df: pd.DataFrame) -> int:
    """
    Count the number of seemless cell handovers (cell to cell switches) for user equipment (UE) based
    on cell_id changes between consecutive ticks, excluding switches to 'RLF'.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'ue_id', 'cell_id', and 'tick' columns.

    +--------+---------+------+
    | ue_id  | cell_id | tick |
    +========+=========+======+
    |   0    |    4    |  1   |
    |   1    |    2    |  2   |
    |   2    |    5    |  3   |
    |   3    |    3    |  4   |
    +--------+---------+------+

    Returns:
        int: Total number of valid cell switches across all UEs.
    """
    count = 0
    df = df.sort_values(by=["ue_id", "tick"])  # Ensure correct order
    prev_cells = {}
    prev_ticks = {}

    for _, row in df.iterrows():
        ue_id, cell_id, tick = row["ue_id"], row["cell_id"], row["tick"]

        if ue_id in prev_cells and prev_cells[ue_id] != cell_id and prev_cells[ue_id] is not None:
            if tick == prev_ticks[ue_id] + 1 and cell_id != "RLF":
                count += 1

        prev_cells[ue_id] = cell_id
        prev_ticks[ue_id] = tick
    return count


def reattach_columns(predicted_df, full_prediction_df):
    # Filter full_prediction_df for the needed columns and drop duplicates based on loc_x and loc_y
    filtered_full_df = full_prediction_df[["mock_ue_id", "tick", "loc_x", "loc_y"]].drop_duplicates(
        subset=["loc_x", "loc_y"]
    )

    # Merge with predicted_df based on loc_x and loc_y, ensuring size matches predicted_df
    merged_df = pd.merge(predicted_df, filtered_full_df, on=["loc_x", "loc_y"], how="left")

    # Rename mock_ue_id to ue_id
    merged_df.rename(columns={"mock_ue_id": "ue_id"}, inplace=True)

    return merged_df


def calculate_mro_metric(data: pd.DataFrame) -> float:
    """
    Calculated total operational cellular time remaining after loss due to cell handovers (including RLF)

    Parameters:
        data (pd.DataFrame): DataFrame containing UE data with a 'tick' column.

    +--------+---------+------+
    | ue_id  | cell_id | tick |
    +========+=========+======+
    |   0    |    4    |  1   |
    |   1    |    2    |  2   |
    |   2    |    5    |  3   |
    |   3    |    3    |  4   |
    +--------+---------+------+

    Returns:
        float: Effective operational score symbolizing time effectively after subtracting handover and RLF delays.
    """
    # Constants for interruption times
    ts = 50 / 1000  # Convert ms to seconds
    t_nas = 1000 / 1000  # Convert ms to seconds

    # Calculate total time (T) based on ticks; assuming each tick represents a uniform time slice
    # This could be adjusted if ticks represent variable time slices
    # Rather than passing the UE Data as whole we can send just an integar for tick
    ticks = len(data["tick"].unique())
    # Assuming each tick represents 50ms (this value may need to be adjusted based on actual data characteristics)
    tick_duration_seconds = 1  # 1 second per tick
    T = ticks * tick_duration_seconds

    ns_handover_count = _count_handovers(data)  # Count of handovers to different cells (excluding RLF)
    nf_handover_count = _count_rlf(data)  # Count of handovers to RLF

    # Calculate D
    D = T - (ns_handover_count * ts + nf_handover_count * t_nas)

    return D


def _count_rlf(df: pd.DataFrame) -> int:
    """
    Counts the number of Radio Link Failures (RLF) by analyzing cell handovers onto RLF for UE

    Parameters:
        df (pd.DataFrame): DataFrame containing 'ue_id', 'cell_id', and 'tick' columns.

    +--------+---------+------+
    | ue_id  | cell_id | tick |
    +========+=========+======+
    |   0    |    4    |  1   |
    |   1    |    2    |  2   |
    |   2    |    5    |  3   |
    |   3    |    3    |  4   |
    +--------+---------+------+

    Returns:
        int: Total number of UE transitions to RLF cells.
    """
    return (df["cell_id"] == "RLF").sum()
