import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from notebooks.radp_library import get_ue_data
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.digital_twin.utils.cell_selection import find_hyst_diff, perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD

from .mobility_robustness_optimization import MobilityRobustnessOptimization, calculate_mro_metric


class ReinforcedMRO(MobilityRobustnessOptimization):
    """
    Solves the mobility robustness optimization problem using reinforcement learning (PPO).
    """

    def __init__(
        self,
        mobility_model_params: dict[str, dict],
        topology: pd.DataFrame,
        bdt: Optional[dict[str, BayesianDigitalTwin]] = None,
    ):
        super().__init__(mobility_model_params, topology, bdt)
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger(__name__)
        # will be filled during solve()
        self.score: pd.DataFrame = pd.DataFrame(columns=["hyst", "ttt", "score"])

    def solve(
        self,
        n_epochs: int = 100,
        n_steps: int = 64,
        batch_size: int = 32,
        verbose: int = 0,
    ) -> Tuple[float, int, pd.DataFrame]:
        """
        Trains a PPO agent to optimize hysteresis and TTT values.

        Returns
        -------
        best_hyst : float
        best_ttt  : int
        score_df  : pd.DataFrame
            Columns: ['hyst','ttt','score'] for each attempt during training.
        """
        if not self.bayesian_digital_twins:
            raise ValueError("Bayesian Digital Twins are not trained. Train the models before calculating metrics.")

        total_timesteps = n_epochs * n_steps

        # Load and prepare simulation data
        self.simulation_data = get_ue_data(self.mobility_model_params)
        self.simulation_data = self.simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})

        if self.topology["cell_id"].dtype == int:
            self.topology["cell_id"] = self.topology["cell_id"].apply(lambda x: f"cell_{int(x)}")

        _, full_prediction_df = self._predictions(self.simulation_data)
        self.simulation_data = self._preprocess_simulation_data(full_prediction_df)

        # Define parameter ranges
        max_diff = find_hyst_diff(self.simulation_data)
        num_ticks = self.simulation_data["tick"].nunique()
        hyst_range = [0, max_diff]
        ttt_range = [2, num_ticks + 1]

        # Create and vectorize RL environment
        def make_env():
            return ReinforcedMROEnv(self.simulation_data, RLF_THRESHOLD, hyst_range, ttt_range, verbose=verbose)

        env = DummyVecEnv([make_env])

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # PPO agent
        model = PPO("MlpPolicy", env, verbose=verbose, n_steps=n_steps, batch_size=batch_size, device=device)
        model.learn(total_timesteps)

        # Grab the underlying env to pull logs and best action
        base_env: ReinforcedMROEnv = env.envs[0]

        # Fill self.score
        self.score = pd.DataFrame(base_env.attempts, columns=["hyst", "ttt", "score"])

        best_hyst = float(base_env.best_hyst)
        best_ttt = int(base_env.best_ttt)

        if verbose > 0:
            self.logger.info(
                f"\nOptimized Hyst (best seen): {best_hyst}, "
                f"Optimized TTT (best seen): {best_ttt}, "
                f"Total attempts logged: {len(self.score)}"
            )

        return best_hyst, best_ttt, self.score.copy()


class ReinforcedMROEnv(Env):
    def __init__(self, df, rlf_threshold, hyst_range, ttt_range, verbose=1):
        super().__init__()
        self.df = df
        self.rlf_threshold = rlf_threshold
        self.hyst_range = hyst_range
        self.ttt_range = ttt_range
        self.verbose = verbose

        self.action_space = Box(
            low=np.array([hyst_range[0], ttt_range[0]]),
            high=np.array([hyst_range[1], ttt_range[1]]),
            dtype=np.float64,
        )
        self.observation_space = Box(
            low=np.array([0, 0, 0, self.hyst_range[0], self.ttt_range[0]]),
            high=np.array([1, 1e6, 1e6, self.hyst_range[1], self.ttt_range[1]]),
            dtype=np.float64,
        )

        self.state = np.array([0.0, 0.0, 0.0, 0.0, 2])
        self.current_step = 0
        self.max_steps = 20
        self.episode_num = 1
        self.episode_reward = 0.0

        # Attempt logging
        self.attempts = []  # rows: [hyst, ttt, reward]

        # Track best observed action
        self.best_reward = -np.inf
        self.best_hyst = 0.0
        self.best_ttt = 2

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger(__name__)

    def step(self, action):
        hyst, ttt = action
        ttt = int(round(float(ttt)))
        hyst = float(hyst)

        attached_df = perform_attachment_hyst_ttt(self.df, hyst, ttt, self.rlf_threshold)
        mro_metric, _, _ = calculate_mro_metric(attached_df)

        reward = float(mro_metric)
        self.episode_reward += reward
        self.state = np.array([reward, 0.0, 0.0, hyst, float(ttt)], dtype=np.float64)
        self.current_step += 1

        # Log attempt
        self.attempts.append([hyst, ttt, reward])

        # Track best
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_hyst = hyst
            self.best_ttt = ttt

        terminated = self.current_step >= self.max_steps
        truncated = False

        if self.verbose > 0:
            self.logger.info(
                f"Step {self.current_step} | Hyst: {hyst:.4f}, TTT: {ttt}, Reward: {reward:.6f}, Done: {terminated}"
            )

        if terminated:
            if self.verbose > 0:
                avg_reward = self.episode_reward / self.max_steps
                self.logger.info(f"Episode {self.episode_num} average reward: {avg_reward:.6f}\n")
            self.episode_num += 1
            self.episode_reward = 0.0

        return self.state, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 2], dtype=np.float64)
        self.current_step = 0
        return self.state, {}

    def render(self):
        if self.verbose > 0:
            self.logger.info(f"Current State: {self.state}, Current Step: {self.current_step}")
