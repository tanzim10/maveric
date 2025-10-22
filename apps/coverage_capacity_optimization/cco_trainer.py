# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Coverage and Capacity Optimization (CCO) Training Module.

This module implements training logic for CCO optimization using the dGPCO algorithm.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from apps.coverage_capacity_optimization.cco_env import CCOEnvironment
from apps.coverage_capacity_optimization.constants import CELL_EL_DEG, CELL_ID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CCOTrainer:
    """
    Trainer for Coverage and Capacity Optimization using dGPCO algorithm.
    """
    
    def __init__(self, environment: CCOEnvironment):
        """
        Initialize CCO Trainer.
        
        Args:
            environment: CCO Environment instance
        """
        self.environment = environment
        self.training_history: List[Dict[str, Any]] = []
        
        # Training parameters
        self.num_epochs = 100
        self.epsilon = 0.1  # For epsilon-greedy exploration
        self.seed = 42
        
        # Metrics tracking
        self.current_epoch = 0
        self.best_reward = -np.inf
        self.best_config = None
        
        logger.info("Initialized CCO Trainer")
    
    def train(self, num_epochs: int = 100, epsilon: float = 0.1, seed: int = 42,
              lambda_: float = 0.5, weak_coverage_threshold: float = -90,
              over_coverage_threshold: float = 0) -> Dict[str, Any]:
        """
        Train the CCO optimization model using dGPCO algorithm.
        
        Args:
            num_epochs: Number of training epochs
            epsilon: Exploration rate for epsilon-greedy
            seed: Random seed
            lambda_: Weight parameter for weak vs over coverage
            weak_coverage_threshold: Threshold for weak coverage
            over_coverage_threshold: Threshold for over coverage
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting dGPCO training for {num_epochs} epochs")
        
        # Set training parameters
        self.num_epochs = num_epochs
        self.epsilon = epsilon
        np.random.seed(seed)
        
        # Initialize training
        epoch_rewards = []
        
        # Calculate initial metric
        _, _, initial_reward = self.environment.calc_metric(
            lambda_=lambda_,
            weak_coverage_threshold=weak_coverage_threshold,
            over_coverage_threshold=over_coverage_threshold,
        )
        epoch_rewards.append(initial_reward)
        
        logger.info(f"Before dGPCO starts, CCO metric = {initial_reward:.3f}")
        
        # Save original configuration for reference
        original_config = self.environment.config.copy()
        continuously_unchanged = 0
        
        # Run dGPCO training with round-robin cell selection
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Round-robin cell selection (matches dgpco_cco.py)
            cell_idx = (epoch - 1) % self.environment.num_cells
            
            # Run one epoch of dGPCO for the selected cell
            epoch_reward, changed = self._run_dgpco_epoch(
                cell_idx=cell_idx,
                original_config=original_config,
                lambda_=lambda_,
                weak_coverage_threshold=weak_coverage_threshold,
                over_coverage_threshold=over_coverage_threshold,
            )
            epoch_rewards.append(epoch_reward)
            
            # Track convergence
            if changed:
                continuously_unchanged = 0
            else:
                continuously_unchanged += 1
            
            # Update best configuration
            if epoch_reward > self.best_reward:
                self.best_reward = epoch_reward
                self.best_config = self.environment.config.copy()
            
            # Store training history
            self.training_history.append({
                'epoch': epoch,
                'reward': epoch_reward,
                'config': self.environment.config.copy(),
                'cell_idx': cell_idx,
                'changed': changed
            })
            
            # Check for convergence (matches dgpco_cco.py logic)
            if continuously_unchanged == self.environment.num_cells:
                logger.info(
                    "\nNo change for any cell after 1 full round robin (optimization converged)...."
                    f"\nExiting early at epoch {epoch}..."
                )
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: CCO metric = {epoch_reward:.3f}")
        
        # Final results
        results = {
            'algorithm': 'dgpco',
            'total_epochs': len(epoch_rewards),
            'initial_reward': initial_reward,
            'final_reward': epoch_rewards[-1],
            'best_reward': self.best_reward,
            'average_reward': np.mean(epoch_rewards),
            'training_history': self.training_history,
            'best_config': self.best_config,
            'epoch_rewards': epoch_rewards,
            'converged_early': continuously_unchanged == self.environment.num_cells
        }
        
        logger.info(f"Training completed. Best reward: {self.best_reward:.3f}")
        return results
    
    def _run_dgpco_epoch(self, cell_idx: int, original_config: pd.DataFrame, 
                        lambda_: float, weak_coverage_threshold: float,
                        over_coverage_threshold: float) -> Tuple[float, bool]:
        """
        Run one epoch of dGPCO algorithm for a single cell (matches dgpco_cco.py).
        
        Args:
            cell_idx: Index of the cell to optimize
            original_config: Original configuration for reference
            lambda_: Weight parameter
            weak_coverage_threshold: Weak coverage threshold
            over_coverage_threshold: Over coverage threshold
            
        Returns:
            Tuple of (epoch_reward, changed_flag)
        """
        # Get current cell information
        cell_id = self.environment.config.iloc[cell_idx][CELL_ID]
        current_tilt = self.environment.get_cell_config_by_index(cell_idx, CELL_EL_DEG)
        original_tilt = original_config.iloc[cell_idx][CELL_EL_DEG]
        
        logger.info(f"\nIn epoch: {self.current_epoch:02}/{self.num_epochs}...")
        
        # Calculate current metric
        _, _, current_reward = self.environment.calc_metric(
            lambda_=lambda_,
            weak_coverage_threshold=weak_coverage_threshold,
            over_coverage_threshold=over_coverage_threshold,
        )
        
        # Define tilt adjustments (matches dgpco_cco.py)
        opt_delta = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        rewards = []
        tilts_tried = []
        
        # Get valid tilt values
        valid_tilts = self.environment.get_valid_tilt_values()
        if not valid_tilts:
            logger.warning("No valid tilt values available")
            return current_reward, False
        
        # Find current tilt index in valid values
        try:
            current_tilt_idx = valid_tilts.index(current_tilt)
        except ValueError:
            logger.warning(f"Current tilt {current_tilt} not in valid values")
            return current_reward, False
        
        # Try different tilt adjustments
        for delta in opt_delta:
            new_tilt_idx = current_tilt_idx + delta
            
            # Check bounds
            if new_tilt_idx < 0 or new_tilt_idx >= len(valid_tilts):
                continue
            
            new_tilt = valid_tilts[new_tilt_idx]
            
            # Skip if same as current tilt
            if new_tilt == current_tilt:
                continue
            
            # Temporarily update configuration
            original_tilt_value = self.environment.get_cell_config_by_index(cell_idx, CELL_EL_DEG)
            self.environment.update_cell_config_by_index(cell_idx, CELL_EL_DEG, new_tilt)
            
            # Calculate reward for new tilt
            _, _, reward = self.environment.calc_metric(
                lambda_=lambda_,
                weak_coverage_threshold=weak_coverage_threshold,
                over_coverage_threshold=over_coverage_threshold,
            )
            
            rewards.append(reward)
            tilts_tried.append(new_tilt)
            
            # Restore original tilt
            self.environment.update_cell_config_by_index(cell_idx, CELL_EL_DEG, original_tilt_value)
        
        # Choose best tilt using epsilon-greedy (FIXED LOGIC)
        if tilts_tried:
            if np.random.uniform() < self.epsilon:
                # Random exploration (matches dgpco_cco.py)
                best_idx = np.random.randint(0, len(tilts_tried))
            else:
                # Pick best (matches dgpco_cco.py)
                best_idx = int(np.argmax(rewards))
            
            best_tilt = tilts_tried[best_idx]
            best_reward = rewards[best_idx]
            
            # If current is better than best tried, keep current
            if current_reward >= best_reward:
                best_tilt = current_tilt
                best_reward = current_reward
                changed = False
            else:
                changed = True
            
            # Apply the best tilt
            self.environment.update_cell_config_by_index(cell_idx, CELL_EL_DEG, best_tilt)
            
            # Log the decision (matches dgpco_cco.py format)
            logger.info(
                f"...cell_id={cell_id}, orig_el_deg={original_tilt}, cur_el_deg={current_tilt}, "
                f"elevs_tried={tilts_tried}, best_el_tried={tilts_tried[best_idx]}, best_el={best_tilt}"
            )
            
            if changed:
                logger.info(
                    f"...changing elevation tilt for cell {cell_id} "
                    f"from {current_tilt} to {best_tilt}, to achieve new "
                    f"CCO metric = {best_reward:.3f}"
                )
            else:
                logger.info(f"...keeping same tilt for cell {cell_id} at {current_tilt}")
        else:
            # No valid tilts to try
            best_reward = current_reward
            changed = False
            logger.info(f"...no valid tilt adjustments for cell {cell_id}")
        
        return best_reward, changed
    
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        import pickle
        
        model_data = {
            'best_config': self.best_config,
            'best_reward': self.best_reward,
            'training_history': self.training_history,
            'algorithm': 'dgpco'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from file.
        
        Args:
            filepath: Path to model file
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_config = model_data.get('best_config')
        self.best_reward = model_data.get('best_reward', -np.inf)
        self.training_history = model_data.get('training_history', [])
        
        logger.info(f"Model loaded from: {filepath}")
    
    def get_training_summary(self) -> str:
        """
        Get training summary as string.
        
        Returns:
            Training summary
        """
        if not self.training_history:
            return "No training history available"
        
        summary = f"""
Training Summary:
  Algorithm: dGPCO
  Total Epochs: {len(self.training_history)}
  Best Reward: {self.best_reward:.3f}
  Final Reward: {self.training_history[-1]['reward']:.3f}
"""
        
        return summary