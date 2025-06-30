import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

try:
    from stable_baselines3 import PPO
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
    from radp.digital_twin.utils.cell_selection import perform_attachment
except ImportError as e:
    print(f"FATAL: Error importing libraries: {e}."); sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

class EnergySavingVisualizer:
    """
    Visualizes network performance, comparing a baseline vs. an RL-optimized configuration.
    """
    def __init__(self, bdt_model_path: str, rl_model_path: str, topology_path: str, config_path: str, base_ue_data_dir: str):
        self.base_ue_data_dir = base_ue_data_dir
        
        topology_df = pd.read_csv(topology_path)
        config_df = pd.read_csv(config_path)
        self.site_config_df_base = pd.merge(topology_df, config_df, on='cell_id', how='left')
        self.site_config_df_base['cell_el_deg'].fillna(TILT_SET[len(TILT_SET)//2], inplace=True)
        
        required_cols = {'hTx': 25.0, 'hRx': 1.5, 'cell_az_deg': 0.0, 'cell_carrier_freq_mhz': 2100.0}
        for col, val in required_cols.items():
            if col not in self.site_config_df_base.columns:
                self.site_config_df_base[col] = val
        
        logger.info("Loading BDT model map..."); self.bdt_model_map = BayesianDigitalTwin.load_model_map_from_pickle(bdt_model_path)
        logger.info(f"Loaded BDT map for {len(self.bdt_model_map)} cells.")

        logger.info("Loading RL agent..."); rl_model_path_zip = rl_model_path if rl_model_path.endswith(".zip") else f"{rl_model_path}.zip"
        self.rl_model = PPO.load(rl_model_path_zip)
        logger.info("RL agent loaded.")
        
        self.COL_LON = getattr(c, 'LOC_X', 'loc_x'); self.COL_LAT = getattr(c, 'LOC_Y', 'loc_y')
        self.COL_CELL_LON = getattr(c, 'CELL_LON', 'cell_lon'); self.COL_CELL_LAT = getattr(c, 'CELL_LAT', 'cell_lat')
        self.COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id'); self.COL_UE_ID = 'mock_ue_id' # Aligned with generated data
        self.COL_RXPOWER_DBM = getattr(c, 'RXPOWER_DBM', 'rxpower_dbm'); self.COL_RSRP_DBM = getattr(c, 'RSRP_DBM', 'rsrp_dbm')

        # Define colors for UE visualization (inner circle)
        self.ue_colors_rgb = [
            (30, 144, 255),   # DodgerBlue
            (255, 215, 0),    # Gold
            (138, 43, 226),   # BlueViolet
            (255, 69, 0),     # OrangeRed
            (50, 205, 50),    # LimeGreen
            (255, 20, 147),   # DeepPink
            (0, 191, 255),    # DeepSkyBlue
            (255, 140, 0),    # DarkOrange
            (148, 0, 211),    # DarkViolet
            (34, 139, 34),    # ForestGreen
        ]
        self.ue_colors = [(r/255, g/255, b/255) for r, g, b in self.ue_colors_rgb]

    def _run_local_simulation(self, ue_data: pd.DataFrame, site_config: pd.DataFrame) -> pd.DataFrame:
        """Runs a local RF simulation and performs cell attachment."""
        all_preds_list = []
        active_cell_ids = site_config[self.COL_CELL_ID].unique()
        for cell_id in active_cell_ids:
            bdt_predictor = self.bdt_model_map.get(cell_id)
            if not bdt_predictor: continue
            
            cell_cfg_df = site_config[site_config[self.COL_CELL_ID] == cell_id]
            pred_frames = BayesianDigitalTwin.create_prediction_frames(site_config_df=cell_cfg_df, prediction_frame_template=ue_data)
            df_for_pred = pred_frames.get(cell_id)
            if df_for_pred is not None and not df_for_pred.empty:
                bdt_predictor.predict_distributed_gpmodel(prediction_dfs=[df_for_pred])
                all_preds_list.append(df_for_pred)
        
        if not all_preds_list: return pd.DataFrame()
        combined_preds = pd.concat(all_preds_list, ignore_index=True)
        return perform_attachment(combined_preds, site_config)

    def _create_dynamic_circle_tower(self, ax, center_lon, center_lat, site_cells_info, radius=0.003):
        """
        Creates a dynamic concentric circle to represent a tower.
        
        Args:
            ax: matplotlib axis
            center_lon, center_lat: tower coordinates
            site_cells_info: list of dict with keys: 'cell_id', 'active', 'ue_color_idx'
            radius: circle radius in coordinate units
        """
        num_cells = len(site_cells_info)
        if num_cells == 0:
            return
            
        # Calculate angles for each cell segment
        angle_per_cell = 360.0 / num_cells
        start_angle = 90.0  # Start from top
        
        patches = []
        
        for i, cell_info in enumerate(site_cells_info):
            # Calculate angles for this cell's segment
            segment_start = start_angle - (i * angle_per_cell)
            segment_end = segment_start - angle_per_cell
            
            # Outer ring - cell status (active/inactive)
            outer_color = 'green' if cell_info['active'] else 'red'
            outer_wedge = Wedge(
                center=(center_lon, center_lat),
                r=radius,
                theta1=segment_end,
                theta2=segment_start,
                width=radius * 0.3,  # 30% of radius for outer ring
                facecolor=outer_color,
                edgecolor='black',
                linewidth=0.5
            )
            patches.append(outer_wedge)
            
            # Inner circle - UE attachment (if cell is active)
            if cell_info['active'] and cell_info['ue_color_idx'] is not None:
                # Ensure color index is within bounds
                safe_color_idx = cell_info['ue_color_idx'] % len(self.ue_colors)
                inner_color = self.ue_colors[safe_color_idx]
                inner_wedge = Wedge(
                    center=(center_lon, center_lat),
                    r=radius * 0.7,  # 70% of radius for inner circle
                    theta1=segment_end,
                    theta2=segment_start,
                    width=radius * 0.7,  # Full width to center
                    facecolor=inner_color,
                    edgecolor='black',
                    linewidth=0.3
                )
                patches.append(inner_wedge)
            elif cell_info['active']:
                # Active but no UEs - show as light gray
                inner_wedge = Wedge(
                    center=(center_lon, center_lat),
                    r=radius * 0.7,
                    theta1=segment_end,
                    theta2=segment_start,
                    width=radius * 0.7,
                    facecolor='lightgray',
                    edgecolor='black',
                    linewidth=0.3
                )
                patches.append(inner_wedge)
        
        # Add all patches to the axis
        for patch in patches:
            ax.add_patch(patch)

    def _get_ue_color_mapping(self, attached_data: pd.DataFrame) -> Dict[str, int]:
        """Create a mapping from cell_id to UE color index based on serving UEs."""
        if attached_data.empty:
            return {}
            
        # Group UEs by serving cell and assign color indices
        cell_color_map = {}
        unique_cells = attached_data[self.COL_CELL_ID].unique()
        
        for i, cell_id in enumerate(sorted(unique_cells)):
            # Ensure color index is within bounds using modulo
            cell_color_map[cell_id] = i % len(self.ue_colors)
            
        return cell_color_map

    def _plot_scenario(self, ax, title: str, ue_data: pd.DataFrame, attached_data: pd.DataFrame,
                       active_cell_ids: set, site_id_col: str):
        """Helper to generate a single subplot with dynamic circle towers."""
        ax.set_title(title, fontsize=16)
        
        # Plot UEs
        merge_keys = [self.COL_LON, self.COL_LAT]
        plot_df = pd.merge(ue_data, attached_data, on=merge_keys, how="left")
        plot_df.rename(columns={self.COL_CELL_ID: 'serving_cell_id'}, inplace=True)
        
        served_ues = plot_df.dropna(subset=['serving_cell_id'])
        unique_cells = sorted(served_ues["serving_cell_id"].unique()) if not served_ues.empty else []
        
        # Get UE color mapping
        ue_color_map = self._get_ue_color_mapping(attached_data)
        
        # Plot served UEs with colors matching their serving cells
        if unique_cells:
            for i, cell_id in enumerate(unique_cells):
                cell_ues = served_ues[served_ues["serving_cell_id"] == cell_id]
                color_idx = ue_color_map.get(cell_id, i % len(self.ue_colors))
                # Ensure color_idx is within bounds
                color_idx = color_idx % len(self.ue_colors)
                ax.scatter(
                    cell_ues[self.COL_LON], 
                    cell_ues[self.COL_LAT], 
                    color=self.ue_colors[color_idx], 
                    s=10, 
                    alpha=0.8, 
                    label=f"UEs ({cell_id})"
                )

        # Plot disconnected UEs
        no_serve_ues = plot_df[plot_df['serving_cell_id'].isna()]
        if not no_serve_ues.empty:
            ax.scatter(
                no_serve_ues[self.COL_LON], 
                no_serve_ues[self.COL_LAT], 
                c='darkorange', 
                marker='x', 
                s=25, 
                label='Disconnected UEs'
            )

        # Create dynamic circle towers
        site_locations = self.site_config_df_base.drop_duplicates(subset=[site_id_col]).copy()
        
        for _, site_row in site_locations.iterrows():
            site_id = site_row[site_id_col]
            site_cells = self.site_config_df_base[self.site_config_df_base[site_id_col] == site_id]
            
            # Prepare cell information for this site
            site_cells_info = []
            for _, cell_row in site_cells.iterrows():
                cell_id = cell_row[self.COL_CELL_ID]
                is_active = cell_id in active_cell_ids
                
                # Get UE color index if this cell is serving UEs
                ue_color_idx = None
                if is_active and cell_id in ue_color_map:
                    ue_color_idx = ue_color_map[cell_id]
                
                site_cells_info.append({
                    'cell_id': cell_id,
                    'active': is_active,
                    'ue_color_idx': ue_color_idx
                })
            
            # Create the dynamic circle tower
            self._create_dynamic_circle_tower(
                ax, 
                site_row[self.COL_CELL_LON], 
                site_row[self.COL_CELL_LAT], 
                site_cells_info
            )
            
            # Add site ID label
            ax.text(
                site_row[self.COL_CELL_LON] + 0.001, 
                site_row[self.COL_CELL_LAT], 
                site_id, 
                fontsize=9, 
                ha='left'
            )

        # Create custom legend
        legend_elements = []
        
        # UE legend entries
        if unique_cells:
            for i, cell_id in enumerate(unique_cells[:5]):  # Limit to first 5 to avoid clutter
                color_idx = ue_color_map.get(cell_id, i % len(self.ue_colors))
                # Ensure color index is within bounds
                safe_color_idx = color_idx % len(self.ue_colors)
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=self.ue_colors[safe_color_idx], 
                           markersize=8, label=f"UEs ({cell_id})")
                )
        
        # Disconnected UEs
        if not no_serve_ues.empty:
            legend_elements.append(
                Line2D([0], [0], marker='x', color='darkorange', 
                       linestyle='None', markersize=8, label='Disconnected UEs')
            )
        
        # Tower status legend
        legend_elements.extend([
            Patch(facecolor='green', edgecolor='black', label='Active Cell (Outer Ring)'),
            Patch(facecolor='red', edgecolor='black', label='Inactive Cell (Outer Ring)'),
            Patch(facecolor='lightgray', edgecolor='black', label='Active Cell, No UEs (Inner)')
        ])

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(handles=legend_elements, loc='best', fontsize='small')

    def generate_comparison_plots(self, day: int, tick: int, output_dir: str):
        ue_data_dir = os.path.join(self.base_ue_data_dir, f"Day_{day}", "ue_data_gym_ready")
        ue_data_file = os.path.join(ue_data_dir, f"generated_ue_data_for_cco_{tick}.csv")
        if not os.path.exists(ue_data_file): 
            logger.error(f"Test UE data file not found: {ue_data_file}"); 
            return
        
        ue_data_df = pd.read_csv(ue_data_file)
        if self.COL_UE_ID not in ue_data_df.columns:
            logger.error(f"UE data file is missing required column: '{self.COL_UE_ID}'")
            return

        def attach_ues(predictions_df: pd.DataFrame):
            if predictions_df.empty or self.COL_UE_ID not in predictions_df.columns:
                return pd.DataFrame()
            idx = predictions_df.groupby(self.COL_UE_ID)[self.COL_RXPOWER_DBM].idxmax()
            serving_data = predictions_df.loc[idx].copy()
            return serving_data

        site_id_col = getattr(c, 'SITE_ID', 'site_id')
        if site_id_col not in self.site_config_df_base.columns:
            self.site_config_df_base['site_id_temp'] = self.site_config_df_base[self.COL_CELL_ID].str.rsplit('_', n=1).str[0]
            site_id_col = 'site_id_temp'

        # --- 1. Baseline Scenario ---
        logger.info("Simulating baseline scenario (initial tilts, all cells on)...")
        baseline_attached_df = self._run_local_simulation(ue_data_df, self.site_config_df_base)
        baseline_active_cells = set(self.site_config_df_base[self.COL_CELL_ID].unique())

        # --- 2. Optimized Scenario ---
        logger.info("Simulating RL-optimized scenario...")
        action_indices, _ = self.rl_model.predict(tick, deterministic=True)
        
        optimized_active_cells = set()
        for i, cell_action_idx in enumerate(action_indices):
            if cell_action_idx < len(TILT_SET):
                optimized_active_cells.add(self.site_config_df_base[self.COL_CELL_ID].iloc[i])
        
        active_topology_for_sim = self.site_config_df_base[
            self.site_config_df_base[self.COL_CELL_ID].isin(optimized_active_cells)
        ]
        optimized_attached_df = self._run_local_simulation(ue_data_df, active_topology_for_sim)

        # --- 3. Generate Plots ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), sharex=True, sharey=True)
        fig.suptitle(f'Energy Saving Comparison for Test Day {day}, Tick {tick}', fontsize=20)

        self._plot_scenario(
            ax1, 'Baseline: Initial Configuration', 
            ue_data_df, baseline_attached_df, 
            baseline_active_cells, site_id_col
        )
        
        self._plot_scenario(
            ax2, 'Optimized: RL Agent Configuration', 
            ue_data_df, optimized_attached_df,
            optimized_active_cells, site_id_col
        )

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"energy_saving_comparison_day{day}_tick_{tick}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"Comparison plot saved to: {output_path}")
        plt.close(fig)