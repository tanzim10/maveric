import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from gpytorch.utils.warnings import NumericalWarning

from notebooks.radp_library import get_ue_data
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.digital_twin.utils.cell_selection import find_hyst_diff, perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD

from .mobility_robustness_optimization import MobilityRobustnessOptimization, calculate_mro_metric


class SimpleMRO(MobilityRobustnessOptimization):
    """
    Iteratively optimizes the cell attachment strategy to find the best MRO metric.
    Currently, 'perform_attachment' has no parameters to optimize, so this function
    will focus on evaluating its current implementation. This setup is ready to be
    expanded for parameter optimization in future developments.
    """

    def __init__(
        self,
        mobility_model_params: Dict[str, Dict[str, Any]],
        topology: pd.DataFrame,
        bdt: Optional[Dict[str, BayesianDigitalTwin]] = None,
    ):
        # Set up logging first
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger(__name__)
        super().__init__(mobility_model_params, topology, bdt)

    def solve(self, n_epochs=100, verbose=1):
        """
        Solve the mobility robustness optimization problem.
        """
        # Ensure Bayesian Digital Twins are trained before proceeding
        if not self.bayesian_digital_twins:
            raise ValueError("Bayesian Digital Twins are not trained. Train the models before calculating metrics.")

        # Generate and preprocess simulation data
        self.simulation_data = get_ue_data(self.mobility_model_params)
        self.simulation_data = self.simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})

        if self.topology["cell_id"].dtype == int:
            self.topology["cell_id"] = self.topology["cell_id"].apply(lambda x: f"cell_{int(x)}")

        # Predict power and perform attachment
        _, full_prediction_df = self._predictions(self.simulation_data)
        self.simulation_data = full_prediction_df
        self.simulation_data = self._preprocess_simulation_data(self.simulation_data)

        epochs = n_epochs
        hyst = 0.01
        ttt = 5
        rlf_threshold = RLF_THRESHOLD

        attached_df = perform_attachment_hyst_ttt(self.simulation_data, hyst, ttt, rlf_threshold)
        max_diff = find_hyst_diff(self.simulation_data)
        num_ticks = self.simulation_data["tick"].nunique()
        hyst_range = [0, max_diff]
        ttt_range = [2, num_ticks + 1]

        # Suppress the specific NumericalWarning from gpytorch
        warnings.filterwarnings("ignore", category=NumericalWarning)

        self.score = pd.DataFrame(columns=["hyst", "ttt", "score"])

        if verbose == 1:
            header = f"{'Epoch':<6} {'Hyst':<14} {'TTT':<6} {'MRO Metric':<12}"
            self.logger.info(header)
            self.logger.info("-" * len(header))

        # Perform initial MRO metric calculation
        attached_df = perform_attachment_hyst_ttt(self.simulation_data, hyst, ttt, rlf_threshold)
        mro_metric, _, _ = calculate_mro_metric(attached_df)
        self.score.loc[len(self.score)] = [hyst, ttt, mro_metric]

        # Parallelize the MRO metric calculation across epochs
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(epochs):
                while True:
                    hyst = np.random.uniform(hyst_range[0], hyst_range[1])
                    ttt = np.random.randint(ttt_range[0], ttt_range[1])
                    if ttt not in self.score["ttt"].values or hyst not in self.score["hyst"].values:
                        break
                # Submit task for parallel processing
                futures.append(executor.submit(self._calculate_epoch_mro, hyst, ttt))

            # Collect the results
            for future in futures:
                hyst, ttt, mro_metric = future.result()
                self.score.loc[len(self.score)] = [hyst, ttt, mro_metric]

        if verbose == 1:
            self.logger.info(f"\nOptimized Hyst: {self.score.loc[self.score['score'].idxmax(), 'hyst']},")
            self.logger.info(f"\nOptimized TTT: {int(self.score.loc[self.score['score'].idxmax(), 'ttt'])}")

        return (
            self.score.loc[self.score["score"].idxmax(), "hyst"],
            int(self.score.loc[self.score["score"].idxmax(), "ttt"]),
            self.score,
        )

    def _calculate_epoch_mro(self, hyst, ttt):
        """
        Calculate MRO metric for a given hyst and ttt values. This will run in parallel for each epoch.
        """
        attached_df = perform_attachment_hyst_ttt(self.simulation_data, hyst, ttt, RLF_THRESHOLD)
        mro_metric, _, _ = calculate_mro_metric(attached_df)
        self.logger.info(f"Calculated MRO Metric: Hyst = {hyst}, TTT = {ttt}, Score = {mro_metric}")
        return hyst, ttt, mro_metric
