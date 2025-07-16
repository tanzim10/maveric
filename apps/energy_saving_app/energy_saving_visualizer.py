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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

        # Define site base colors 
        self.site_base_colors = [
            (30, 144, 255),   # DodgerBlue - Blue (high contrast)
            (138, 43, 226),   # BlueViolet - Purple (high contrast)
            (255, 69, 0),     # OrangeRed - Red (prioritized, distinct from outer ring red)
            (50, 205, 50),    # LimeGreen - Green (prioritized, distinct from outer ring green)
            (255, 215, 0),    # Gold - Yellow (high contrast, opposite purple)
            (255, 20, 147),   # DeepPink - Pink (medium contrast)
            (0, 191, 255),    # DeepSkyBlue - Light Blue (medium contrast)
            (148, 0, 211),    # DarkViolet - Violet (medium contrast)
            (75, 0, 130),     # Indigo - Dark Purple (medium contrast)
        ]
        
        # Define antenna shades for each site (3 shades for 3 antennas: Base -> Lightest -> Darkest)
        # This provides maximum contrast for the typical 3-antenna configuration
        self.antenna_shades = [1, 1.6, 0.5, 2.0, 0.3, 2.2]  
        
        # Legacy colors for backward compatibility
        self.ue_colors_rgb = self.site_base_colors
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

    def _create_dynamic_circle_tower(self, ax, center_lon, center_lat, site_cells_info, radius=0.002):
        """
        Creates a dynamic concentric circle to represent a tower.
        
        Args:
            ax: matplotlib axis
            center_lon, center_lat: tower coordinates
            site_cells_info: list of dict with keys: 'cell_id', 'active', 'ue_color', 'az_deg'
            radius: circle radius in coordinate units
        """
        num_cells = len(site_cells_info)
        if num_cells == 0:
            return
        
        # Sort site_cells_info by az_deg to align color and wedge assignment
        site_cells_info_sorted = sorted(site_cells_info, key=lambda x: x.get('az_deg', 0.0))
        
        # Calculate dynamic wedge widths and boundaries for this site
        azimuths = [cell_info.get('az_deg', 0.0) for cell_info in site_cells_info_sorted]
        default_width = 360.0 / len(azimuths) 
        wedge_widths = self._calculate_per_antenna_wedge_widths(azimuths, default_width)
        
        # Calculate wedge boundaries using midpoint method
        wedge_boundaries = self._calculate_wedge_boundaries(azimuths)
        
        patches = []
        
        # Detect duplicate azimuths to make their outer rings blue
        duplicate_azimuths = [az for az in set(azimuths) if azimuths.count(az) > 1]
        
        for i, cell_info in enumerate(site_cells_info_sorted):
            # Get azimuth angle for this antenna
            az_deg = cell_info.get('az_deg', 0.0)
            wedge_width = wedge_widths[i]
            
            # Get the calculated boundaries for this antenna using its index
            start_boundary, end_boundary = wedge_boundaries[i]
            
            # Convert boundaries to matplotlib coordinate system
            # Matplotlib: 0° at 3 o'clock, 90° at 12 o'clock
            # Azimuth: 0° at 12 o'clock, 90° at 3 o'clock
            segment_start = 90.0 - start_boundary
            segment_end = 90.0 - end_boundary
            
            logger.debug(f"Cell {cell_info['cell_id']}: az_deg={az_deg}°, boundaries=[{start_boundary}°, {end_boundary}°], segment=[{segment_end}°, {segment_start}°]")
            
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
            
            # Inner circle - UE attachment
            if cell_info['ue_color'] is not None:
                inner_color = cell_info['ue_color']
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
            else:
                # light gray
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

    def _get_site_based_color_mapping(self, attached_data: pd.DataFrame, site_id_col: str) -> Dict[str, tuple]:
        """Create a mapping from cell_id to unique color using a 100-level colormap."""
        if attached_data.empty:
            return {}
            
        # Get site information for all cells
        site_cells = self.site_config_df_base[[self.COL_CELL_ID, site_id_col]].copy()
        
        # Find the maximum number of cells at any site
        max_cells_per_site = site_cells.groupby(site_id_col)[self.COL_CELL_ID].count().max()
        
        # Get unique sites
        unique_sites = site_cells[site_id_col].unique()
        num_sites = len(unique_sites)
        
        # Create color mapping using matplotlib colormap with 100 levels
        import matplotlib.pyplot as plt
        unique_colors = 100
        cmap = plt.get_cmap('tab20', unique_colors)  # Better color separation
        
        # Divide 100 colors into max_cells_per_site buckets
        bucket_size = unique_colors / max_cells_per_site
        
        
        site_step = int(bucket_size/num_sites) # max(int(bucket_size * 0.3), int(bucket_size/num_sites))
        
        cell_color_map = {}
        
        for site_idx, site_id in enumerate(unique_sites):
            site_cells_list = site_cells[site_cells[site_id_col] == site_id][self.COL_CELL_ID].tolist()
            
            for cell_idx, cell_id in enumerate(site_cells_list):
                # Calculate color index: bucket_start + site_offset with modulus for overflow
                bucket_start = cell_idx * bucket_size
                site_offset = (site_idx * 3 % num_sites * site_step) % bucket_size
                color_index = int(bucket_start + site_offset) % unique_colors
                # Ensure we don't exceed the colormap bounds
                
                cell_color_map[cell_id] = cmap(color_index)[:3]
        
        return cell_color_map
    
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

    def _calculate_per_antenna_wedge_widths(self, azimuths: List[float], default_width: float) -> List[float]:
        """
        Calculate wedge widths for each antenna at a site.
        Args:
            azimuths: List of azimuth angles for antennas at the site
            default_width: Default wedge width (360° / antenna_count)
        Returns:
            List of wedge widths for each antenna
        """
        if len(azimuths) <= 1:
            return [360.0]  # Single antenna covers full circle

        # Sort azimuths for easier processing
        sorted_azimuths = sorted(azimuths)
        wedge_widths = []
        
        # Calculate midpoints between adjacent antennas
        midpoints = []
        for i in range(len(sorted_azimuths)):
            current = sorted_azimuths[i]
            next_azimuth = sorted_azimuths[(i + 1) % len(sorted_azimuths)]
            
            # Calculate midpoint
            if next_azimuth == current:
                midpoint = current
            elif next_azimuth > current:
                midpoint = current + (next_azimuth - current) / 2
            else:
                # Handle wrap-around (e.g., 350° to 10°)
                midpoint = current + (360.0 - current + next_azimuth) / 2
                if midpoint >= 360.0:
                    midpoint -= 360.0
            
            midpoints.append(midpoint)
        
        # Calculate wedge widths based on midpoints
        for i in range(len(sorted_azimuths)):
            # Each wedge spans from the midpoint before it to the midpoint after it
            prev_midpoint = midpoints[i - 1]
            next_midpoint = midpoints[i]
            
            # Calculate wedge width
            if next_midpoint > prev_midpoint:
                wedge_width = next_midpoint - prev_midpoint
            else:
                # Handle wrap-around
                wedge_width = (360.0 - prev_midpoint) + next_midpoint
            
            wedge_widths.append(wedge_width)
        
        return wedge_widths

    def _calculate_wedge_boundaries(self, azimuths: List[float]) -> Dict[float, tuple]:
        """
        Calculate the start and end boundaries for each antenna's wedge using midpoint method.
        
        Args:
            azimuths: List of azimuth angles for antennas at the site
            
        Returns:
            Dictionary mapping azimuth to (start_boundary, end_boundary) tuple
        """
        if len(azimuths) <= 1:
            return {azimuths[0]: (0.0, 360.0)} if azimuths else {}
        
        # Sort azimuths to find true adjacency
        sorted_azimuths = sorted(azimuths)
        boundaries_dict = {}
        
        # Calculate midpoints between adjacent antennas
        midpoints = []
        for i in range(len(sorted_azimuths)):
            current = sorted_azimuths[i]
            next_azimuth = sorted_azimuths[(i + 1) % len(sorted_azimuths)]
            
            # Calculate midpoint
            if next_azimuth == current:
                midpoint = current
            elif next_azimuth > current:
                midpoint = current + (next_azimuth - current) / 2
            else:
                # Handle wrap-around (e.g., 350° to 10°)
                midpoint = current + (360.0 - current + next_azimuth) / 2
                if midpoint >= 360.0:
                    midpoint -= 360.0
            
            midpoints.append(midpoint)
        
        # Calculate boundaries for each antenna
        for i in range(len(sorted_azimuths)):
            # Each wedge spans from the midpoint before it to the midpoint after it
            prev_midpoint = midpoints[i - 1]
            next_midpoint = midpoints[i]
            
            # Store boundaries (start, end) for this antenna using index as key
            boundaries_dict[i] = (prev_midpoint, next_midpoint)
        
        return boundaries_dict

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
        
        # Get site-based color mapping
        site_color_map = self._get_site_based_color_mapping(attached_data, site_id_col)
        
        # Plot served UEs with colors matching their serving cells
        if unique_cells:
            for i, cell_id in enumerate(unique_cells):
                cell_ues = served_ues[served_ues["serving_cell_id"] == cell_id]
                cell_color = site_color_map.get(cell_id, self.ue_colors[i % len(self.ue_colors)])
                ax.scatter(
                    cell_ues[self.COL_LON], 
                    cell_ues[self.COL_LAT], 
                    color=cell_color, 
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

                # Always assign the originally assigned color
                ue_color = site_color_map.get(cell_id, (0.7, 0.85, 1.0))

                # Get azimuth angle for this antenna
                az_deg = cell_row.get('cell_az_deg', 0.0)
                logger.debug(f"Cell {cell_id}: azimuth = {az_deg}°")

                site_cells_info.append({
                    'cell_id': cell_id,
                    'active': is_active,
                    'ue_color': ue_color,
                    'az_deg': az_deg
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
                site_row[self.COL_CELL_LON] + 0.003, 
                site_row[self.COL_CELL_LAT] + 0.001, 
                site_id, 
                fontsize=9, 
                ha='left'
            )

        # Create custom legend
        legend_elements = []
        
        # Single legend entry for all colored dots
        if unique_cells:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor='blue',  # Use a representative color
                                               markersize=8, label="UE connected to cell of\ncorresponding color")
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

