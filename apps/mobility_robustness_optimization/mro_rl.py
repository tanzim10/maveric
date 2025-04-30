from .mobility_robustness_optimization import MobilityRobustnessOptimization, calculate_mro_metric
from radp.digital_twin.utils.cell_selection import (perform_attachment_hyst_ttt, find_hyst_diff)
from notebooks.radp_library import get_ue_data
from radp.digital_twin.utils.constants import RLF_THRESHOLD

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env
from gym.spaces import Box
import numpy as np

class ReinforcedMRO(MobilityRobustnessOptimization):
    """
    Iteratively optimizes the cell attachment strategy to find the best MRO metric.
    Currently, 'perform_attachment' has no parameters to optimize, so this function
    will focus on evaluating its current implementation. This setup is ready to be
    expanded for parameter optimization in future developments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization can be done here if needed
    
    def solve(self):
        """
        Solve the mobility robustness optimization problem using PPO RL.
        """
        # Ensure Bayesian Digital Twins are trained before proceeding
        if not self.bayesian_digital_twins:
            raise ValueError("Bayesian Digital Twins are not trained. Train the models before calculating metrics.")
        
        # Generate and preprocess simulation data
        self.simulation_data = get_ue_data(self.mobility_params)
        self.simulation_data = self.simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})

        # Predict power and preprocess data
        predictions, full_prediction_df = self._predictions(self.simulation_data)
        df = self._preprocess_simulation_data(full_prediction_df)

        # Define ranges for hyst and ttt
        max_diff = find_hyst_diff(df)
        num_ticks = df["tick"].nunique()
        hyst_range = [0, max_diff]
        ttt_range = [2, num_ticks + 1]

        # Create the RL environment
        env = ReinforcedMROEnv(df, RLF_THRESHOLD, hyst_range, ttt_range)
        env = DummyVecEnv([lambda: env])  # Vectorize the environment for stable-baselines3

        # Train the PPO agent
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=100)

        # Use the trained model to find the optimal parameters
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        hyst, ttt = action
        ttt = int(ttt)  # Ensure TTT is an integer

        print(f"\nOptimized Hyst: {hyst}, Optimized TTT: {ttt}")
        return hyst, ttt
        

class ReinforcedMROEnv(Env):
    def __init__(self, df, rlf_threshold, hyst_range, ttt_range):
        super().__init__()
        self.df = df
        self.rlf_threshold = rlf_threshold
        self.hyst_range = hyst_range
        self.ttt_range = ttt_range

        # Define action and observation space
        self.action_space = Box(low=np.array([hyst_range[0], ttt_range[0]]), 
                                high=np.array([hyst_range[1], ttt_range[1]]), 
                                dtype=np.float32)
        self.observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Initial state
        self.state = np.array([0.0])
        self.current_step = 0

    def step(self, action):
        hyst, ttt = action
        ttt = int(ttt)  # Ensure TTT is an integer

        # Perform attachment and calculate MRO Metric
        attached_df = perform_attachment_hyst_ttt(self.df, hyst, ttt, self.rlf_threshold)
        mro_metric = calculate_mro_metric(attached_df)

        # Reward is the MRO metric
        reward = mro_metric

        # Update state (optional, here we keep it simple)
        self.state = np.array([reward])

        # PPO typically doesn't require a terminal state, but you can define one if needed
        done = False

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([0.0])
        self.current_step = 0
        return self.state

    def render(self, mode="human"):
        print(f"Current State: {self.state}, Current Step: {self.current_step}")