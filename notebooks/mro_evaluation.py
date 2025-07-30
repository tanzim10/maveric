import json
import logging
import time

import pandas as pd
from radp_library import calculate_naive_mro_metric, count_handovers, preprocess_ue_data, reattach_columns

from apps.mobility_robustness_optimization.mobility_robustness_optimization import calculate_mro_metric
from apps.mobility_robustness_optimization.mro_ml import BayesianMRO
from apps.mobility_robustness_optimization.mro_rl import ReinforcedMRO
from apps.mobility_robustness_optimization.simple_mro import SimpleMRO
from radp.digital_twin.utils.cell_selection import perform_attachment, perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD


def run_simple_mro(params, topology, data, epochs):
    mro = SimpleMRO(params, topology)
    mro.train_or_update_rf_twins(data)
    return mro.solve(n_epochs=epochs)


def run_xgboost(params, topology, data, epochs):
    mro = BayesianMRO(params, topology, model_type="xgboost")
    mro.train_or_update_rf_twins(data)
    return mro.solve(n_epochs=epochs)


def run_gpr(params, topology, data, epochs):
    mro = BayesianMRO(params, topology)
    mro.train_or_update_rf_twins(data)
    return mro.solve(n_epochs=epochs)


def run_rl_mro(params, topology, data, epochs):
    mro = ReinforcedMRO(params, topology)
    mro.train_or_update_rf_twins(data)
    return mro.solve(total_timesteps=epochs)


def run_naive_attachment(data, topology):
    data = data.rename(columns={"longitude": "loc_x", "latitude": "loc_y", "cell_rxpwr_dbm": "rxpower_dbm"})
    attached_df = perform_attachment(data, topology)
    attached_df = reattach_columns(attached_df, data)

    ns, nf, no_change = count_handovers(attached_df)
    return calculate_naive_mro_metric(ns, nf, data)


def percentage_difference(x: float, y: float) -> str:
    if x == 0:
        return "Undefined (x is zero)"

    diff = ((y - x) / x) * 100
    sign = "+" if diff >= 0 else "-"
    return f"{sign}{abs(diff):.2f}%"


def timed_run(logger, label, func, *args, **kwargs):
    logger.info(f"Finding optimal hysteresis and TTT values for {label}\n")
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    logger.info(f"{label} time: {end - start:.2f} seconds")
    return result


if __name__ == "__main__":
    start_total = time.time()
    with open("notebooks/mro_hyperparams.json", "r") as file:
        hyperparams = json.load(file)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading topology and UE data")

    topology = pd.read_csv(hyperparams["topology"])
    ue_data = pd.read_csv(hyperparams["ue_data"])  # TODO: Change UE

    epochs = hyperparams["epochs"]
    split_ratio = hyperparams["split_ratio"]

    logger.info("Preprocessing data")

    ue_data.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)
    full_data = preprocess_ue_data(ue_data, topology)

    logger.info("Splitting data into training and testing sets")

    unique_ticks = full_data["tick"].sort_values().unique()
    n_ticks = len(unique_ticks)
    split_index = int(n_ticks * split_ratio)

    first_80_ticks = unique_ticks[:split_index]
    last_20_ticks = unique_ticks[split_index:]

    train_data = full_data[full_data["tick"].isin(first_80_ticks)]
    test_data = full_data[full_data["tick"].isin(last_20_ticks)]

    logger.info(f"Training data file size: {train_data.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
    logger.info(f"Number of rows in Training data: {len(train_data)}\n")

    logger.info(f"Testing data file size: {test_data.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
    logger.info(f"Number of rows in Testing data: {len(test_data)}\n")

    params = hyperparams["mobility_model_params"]

    logger.info("Starting The Training Phase:\n")

    s_hyst, s_ttt = timed_run(
        logger, f"Simple MRO on {epochs} epochs", run_simple_mro, params, topology, train_data, epochs
    )
    xgb_hyst, xgb_ttt = timed_run(
        logger, f"XGBoost MRO on {epochs} Epochs", run_xgboost, params, topology, train_data, epochs
    )
    gpr_hyst, gpr_ttt = timed_run(logger, f"GPR MRO on {epochs} Epochs", run_gpr, params, topology, train_data, epochs)
    rl_hyst, rl_ttt = timed_run(
        logger, f"Reinforced MRO on {epochs} Epochs", run_rl_mro, params, topology, train_data, epochs
    )
    base_score = timed_run(logger, "Naive Attachment", run_naive_attachment, train_data, topology)

    logger.info("Preprocessing Testing Data")

    train_data = train_data.rename(columns={"cell_rxpwr_dbm": "cell_rxpower_dbm", "mock_ue_id": "ue_id"})
    test_data = test_data.rename(columns={"cell_rxpwr_dbm": "cell_rxpower_dbm", "mock_ue_id": "ue_id"})

    mro = SimpleMRO(params, topology)
    train_data = mro._add_sinr_column(train_data)
    test_data = mro._add_sinr_column(test_data)

    logger.info("Evaluating Scores\n")

    # Simple MRO
    logger.info("Simple")
    attached_df = perform_attachment_hyst_ttt(train_data, s_hyst, s_ttt, rlf_threshold=RLF_THRESHOLD)
    simple_score_train = calculate_mro_metric(attached_df)
    attached_df = perform_attachment_hyst_ttt(test_data, s_hyst, s_ttt, rlf_threshold=RLF_THRESHOLD)
    simple_score_test = calculate_mro_metric(attached_df)

    # GPR MRO
    logger.info("GPR")
    attached_df = perform_attachment_hyst_ttt(train_data, gpr_hyst, gpr_ttt, rlf_threshold=RLF_THRESHOLD)
    gpr_score_train = calculate_mro_metric(attached_df)
    attached_df = perform_attachment_hyst_ttt(test_data, gpr_hyst, gpr_ttt, rlf_threshold=RLF_THRESHOLD)
    gpr_score_test = calculate_mro_metric(attached_df)

    # Reinforced MRO
    attached_df = perform_attachment_hyst_ttt(train_data, rl_hyst, rl_ttt, rlf_threshold=RLF_THRESHOLD)
    rl_score_train = calculate_mro_metric(attached_df)
    attached_df = perform_attachment_hyst_ttt(test_data, rl_hyst, rl_ttt, rlf_threshold=RLF_THRESHOLD)
    rl_score_test = calculate_mro_metric(attached_df)

    # XGBoost MRO
    logger.info("XGBoost")
    attached_df = perform_attachment_hyst_ttt(train_data, xgb_hyst, xgb_ttt, rlf_threshold=RLF_THRESHOLD)
    xgb_score_train = calculate_mro_metric(attached_df)
    attached_df = perform_attachment_hyst_ttt(test_data, xgb_hyst, xgb_ttt, rlf_threshold=RLF_THRESHOLD)
    xgb_score_test = calculate_mro_metric(attached_df)

    # Base MRO Score
    logger.info("Naive Attachment\n")
    train_data = train_data.rename(columns={"cell_rxpower_dbm": "rxpower_dbm", "ue_id": "mock_ue_id"})
    test_data = test_data.rename(columns={"cell_rxpower_dbm": "rxpower_dbm", "ue_id": "mock_ue_id"})
    train_metric = run_naive_attachment(train_data, topology)
    test_metric = run_naive_attachment(test_data, topology)

    logger.info("Results Summary:")
    logger.info(f"Simple MRO: \t\tHyst = {s_hyst:.3f}, TTT = {s_ttt}")
    logger.info(
        f"  Train Score: \t\t{simple_score_train:.2f} ({percentage_difference(train_metric, simple_score_train)})"
    )
    logger.info(
        f"  Test Score:  \t\t{simple_score_test:.2f} ({percentage_difference(test_metric, simple_score_test)})\n"
    )

    logger.info(f"GPR MRO: \t\tHyst = {gpr_hyst:.3f}, TTT = {gpr_ttt}")
    logger.info(f"  Train Score: \t\t{gpr_score_train:.2f} ({percentage_difference(train_metric, gpr_score_train)})")
    logger.info(f"  Test Score: \t\t{gpr_score_test:.2f} ({percentage_difference(test_metric, gpr_score_test)})\n")

    logger.info(f"XGBoost MRO: \t\tHyst = {xgb_hyst:.3f}, TTT = {xgb_ttt}")
    logger.info(f"  Train Score: \t\t{xgb_score_train:.2f} ({percentage_difference(train_metric, xgb_score_train)})")
    logger.info(f"  Test Score:  \t\t{xgb_score_test:.2f} ({percentage_difference(test_metric, xgb_score_test)})\n")

    logger.info(f"Reinforced MRO: \t\tHyst = {rl_hyst:.3f}, TTT = {rl_ttt}")
    logger.info(f"  Train Score: \t\t{rl_score_train:.2f} ({percentage_difference(train_metric, rl_score_train)})")
    logger.info(f"  Test Score:  \t\t{rl_score_test:.2f} ({percentage_difference(test_metric, rl_score_test)})\n")

    end_total = time.time()
    logger.info(f"Total time elapsed: {end_total - start_total:.2f} seconds")
