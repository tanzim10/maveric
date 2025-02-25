import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# implement mro_score func for perform_attachment_hyst_ttt func before running the code
# mro_score(connected_df: Pandas DataFrame, hyst: float, ttt: int) -> float

class MROEnv(gym.Env):
    def __init__(self, df, max_rxpower_diff, num_ticks):
        super(MROEnv, self).__init__()
        self.df = df
        self.max_rxpower_diff = max_rxpower_diff
        self.num_ticks = num_ticks
        
        # Define Hyst TTT limits
        self.action_space = gym.spaces.Box(low=np.array([0, 2]), high=np.array([max_rxpower_diff, num_ticks]), dtype=np.float32)
        self.observation_space = gym.spaces.Discrete(1)  # No meaningful state, just optimizing params
        
    def step(self, action):
        hyst, ttt = action
        reward = mro_score(self.df, hyst, ttt)  # Call the provided function
        return np.array([0]), reward, False, {}

    def reset(self):
        return np.array([0])
    
    def render(self, mode="human"):
        # add plotting when necessary for debugging
        pass

# Function to train RL model
def train_rl_model(model_type, df, max_rxpower_diff, num_ticks, timesteps=10000):
    env = DummyVecEnv([lambda: MROEnv(df, max_rxpower_diff, num_ticks)])
    
    if model_type == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)
    elif model_type == "DQN":
        model = DQN("MlpPolicy", env, verbose=1)
    else:
        raise ValueError("Unsupported model type. Choose either 'PPO' or 'DQN'.")
    
    model.learn(total_timesteps=timesteps)
    return model

# Example usage
# model = train_rl_model("PPO", df, max_rxpower_diff=10, num_ticks=100, timesteps=50000)
