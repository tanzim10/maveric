import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from notebooks.radp_library import get_ue_data
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.digital_twin.utils.cell_selection import find_hyst_diff, perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD

from .mobility_robustness_optimization import MobilityRobustnessOptimization, calculate_mro_metric


class BayesianMRO(MobilityRobustnessOptimization):
    """Optimize hysteresis and TTT using Bayesian optimization."""

    def __init__(
        self,
        mobility_model_params: Dict[str, Dict[str, Any]],
        topology: pd.DataFrame,
        bdt: Optional[Dict[str, BayesianDigitalTwin]] = None,
        model_type: str = "gpr",
    ):
        # Set up logging first
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger(__name__)
        super().__init__(mobility_model_params, topology, bdt)
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # will be filled by solve()
        self.score: pd.DataFrame = pd.DataFrame(columns=["iter", "hyst", "ttt", "score"])

    def _expected_improvement(self, X: np.ndarray, model, best_y: float) -> np.ndarray:
        if self.model_type == "xgboost":
            y_pred = model.predict(X)
            return y_pred - best_y
        else:
            mu, std = model.predict(X, return_std=True)
            std = np.maximum(std, 1e-9)
            Z = (mu - best_y) / std
            return (mu - best_y) * norm.cdf(Z) + std * norm.pdf(Z)

    def _init_model(self):
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBRegressor  # type: ignore
            except ImportError as e:
                raise ImportError("xgboost is required for model_type='xgboost'") from e
            return XGBRegressor(objective="reg:squarederror")
        else:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(
                noise_level=1e-5, noise_level_bounds=(1e-10, 1e1)
            )
            return GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    def solve(self, n_epochs: int = 20, init_samples: int = 5, verbose: int = 1) -> Tuple[float, int, pd.DataFrame]:
        """
        Runs Bayesian optimization for hysteresis (hyst) and time-to-trigger (ttt).

        Returns
        -------
        best_hyst : float
        best_ttt  : int
        score_df  : pd.DataFrame
            Columns: ['iter', 'hyst', 'ttt', 'score'], each row is an attempted evaluation.
        """
        if init_samples <= 0:
            raise ValueError("init_samples must be > 0")
        if n_epochs < 0:
            raise ValueError("n_epochs must be >= 0")

        if not self.bayesian_digital_twins:
            raise ValueError("Bayesian Digital Twins are not trained. Train the models before calculating metrics.")

        # Prepare simulation data
        self.simulation_data = get_ue_data(self.mobility_model_params)
        self.simulation_data = self.simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})

        if self.topology["cell_id"].dtype == int:
            self.topology["cell_id"] = self.topology["cell_id"].apply(lambda x: f"cell_{int(x)}")

        _, full_prediction_df = self._predictions(self.simulation_data)
        self.simulation_data = self._preprocess_simulation_data(full_prediction_df)

        rlf_threshold = RLF_THRESHOLD
        max_diff = find_hyst_diff(self.simulation_data)
        num_ticks = self.simulation_data["tick"].nunique()
        hyst_range = [0, max_diff]
        ttt_range = [2, num_ticks + 1]

        # reset score DF
        self.score = pd.DataFrame(columns=["iter", "hyst", "ttt", "score"])

        # Initial design
        X, y = [], []
        iter_counter = 0
        for _ in range(init_samples):
            hyst = float(np.random.uniform(hyst_range[0], hyst_range[1]))
            ttt = int(np.random.randint(ttt_range[0], ttt_range[1]))
            attached_df = perform_attachment_hyst_ttt(self.simulation_data, hyst, ttt, rlf_threshold)
            metric, _, _ = calculate_mro_metric(attached_df)

            X.append([hyst, ttt])
            y.append(metric)
            self.score.loc[len(self.score)] = [iter_counter, hyst, ttt, float(metric)]
            iter_counter += 1

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        model = self._init_model()
        best_y = float(y.max())
        best_idx = int(y.argmax())

        # Bayesian optimization loop
        for _ in range(n_epochs):
            model.fit(X, y)

            cand_hyst = np.random.uniform(hyst_range[0], hyst_range[1], size=256)
            cand_ttt = np.random.randint(ttt_range[0], ttt_range[1], size=256)
            candidates = np.column_stack([cand_hyst, cand_ttt])

            scores = self._expected_improvement(candidates, model, best_y)
            idx = int(np.argmax(scores))
            hyst, ttt = float(candidates[idx, 0]), int(round(candidates[idx, 1]))

            attached_df = perform_attachment_hyst_ttt(self.simulation_data, hyst, ttt, rlf_threshold)
            metric, _, _ = calculate_mro_metric(attached_df)

            X = np.vstack([X, [hyst, float(ttt)]])
            y = np.append(y, float(metric))

            self.score.loc[len(self.score)] = [iter_counter, hyst, ttt, float(metric)]
            iter_counter += 1

            if metric > best_y:
                best_y = float(metric)
                best_idx = len(y) - 1

        best_hyst = float(X[best_idx, 0])
        best_ttt = int(round(X[best_idx, 1]))

        if verbose == 1:
            self.logger.info(
                f"\nOptimized Hyst: {best_hyst},\nOptimized TTT: {best_ttt},\nBest Score: {best_y:.6f}, "
                f"Total evals: {len(self.score)}"
            )

        # return a copy so callers can't mutate our internal DF inadvertently
        return best_hyst, best_ttt, self.score.copy()
