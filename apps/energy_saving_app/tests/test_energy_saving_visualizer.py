#!/usr/bin/env python3
"""
Test script for energy saving visualizer with azimuth-aligned wedges.
This script loads the already-trained models and tests the visualization changes.
"""

import os
import sys
import logging

# Add the project root to the path to import the apps
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Force reload the visualizer module to get the latest changes
import importlib
if 'apps.energy_savings.energy_saving_visualizer' in sys.modules:
    importlib.reload(sys.modules['apps.energy_savings.energy_saving_visualizer'])

from apps.energy_saving_app.energy_saving_visualizer import EnergySavingVisualizer
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import pandas as pd

# Set up logging to DEBUG level to see the azimuth calculations
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_azimuth_angles():
    """Test azimuth angle calculations with different angles to demonstrate the functionality."""
    
    # Test data with different azimuth angles
    test_cells = [
        {'cell_id': 'cell_1_0', 'az_deg': 0.0},    # North
        {'cell_id': 'cell_1_1', 'az_deg': 45.0},   # Northeast  
        {'cell_id': 'cell_1_2', 'az_deg': 90.0},   # East
        {'cell_id': 'cell_1_3', 'az_deg': 135.0},  # Southeast
        {'cell_id': 'cell_1_4', 'az_deg': 180.0},  # South
        {'cell_id': 'cell_1_5', 'az_deg': 225.0},  # Southwest
    ]
    
    # Create a simple plot to show the wedges
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    center_lon, center_lat = 0.0, 0.0
    radius = 0.1
    wedge_width = 30.0
    
    for cell_info in test_cells:
        az_deg = cell_info['az_deg']
        
        # Calculate wedge angles based on azimuth direction
        # Azimuth 0° points North, 90° points East, etc.
        # Convert to matplotlib's coordinate system (0° at 3 o'clock, 90° at 12 o'clock)
        center_angle = 90.0 - az_deg  # Convert azimuth to matplotlib angle
        segment_start = center_angle + (wedge_width / 2.0)
        segment_end = center_angle - (wedge_width / 2.0)
        
        logger.info(f"Cell {cell_info['cell_id']}: az_deg={az_deg}°, center_angle={center_angle}°, segment=[{segment_end}°, {segment_start}°]")
        
        # Create wedge
        wedge = Wedge(
            center=(center_lon, center_lat),
            r=radius,
            theta1=segment_end,
            theta2=segment_start,
            width=radius * 0.3,
            facecolor='green',
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(wedge)
        
        # Add text label
        label_angle_rad = (center_angle * 3.14159 / 180.0) - (3.14159 / 2.0)  # Adjust for text positioning
        label_x = center_lon + (radius * 1.2) * np.cos(label_angle_rad)
        label_y = center_lat + (radius * 1.2) * np.sin(label_angle_rad)
        ax.text(label_x, label_y, f"{az_deg}°", ha='center', va='center', fontsize=8)
    
    # Add center point
    ax.scatter([center_lon], [center_lat], c='red', s=100, zorder=10)
    
    # Add cardinal directions
    ax.text(0, radius * 1.5, 'N', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(radius * 1.5, 0, 'E', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0, -radius * 1.5, 'S', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(-radius * 1.5, 0, 'W', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(-radius * 2, radius * 2)
    ax.set_ylim(-radius * 2, radius * 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Azimuth Angle Test - Different Angles')
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'azimuth_angle_test.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    logger.info(f"Azimuth test plot saved to: {output_path}")
    plt.close(fig)


def test_different_azimuth_angles(visualizer, day, tick, output_dir):
    """Test the visualizer with different azimuth angles (non-evenly distributed)."""
    logger.info("=== TESTING DIFFERENT AZIMUTH ANGLES ===")
    
    # Create shallow copy to avoid modifying the original data
    modified_site_config = visualizer.site_config_df_base.copy()
    
    # Create site_id_temp column if it doesn't exist
    if 'site_id_temp' not in modified_site_config.columns:
        modified_site_config['site_id_temp'] = modified_site_config['cell_id'].str.rsplit('_', n=1).str[0]
    
    # Get unique sites
    site_ids = modified_site_config['site_id_temp'].unique()
    
    # Define azimuth configurations for each site by index
    azimuth_configs = [
        [0, 120, 240],      # Site 1: Evenly spaced
        [45, 90, 225],     # Site 2: Non-evenly spaced
        [30, 80, 270],    # Site 3: Another config
        [60, 300, 300],    # Site 4: Duplicate azimuths
        [15, 190, 255],    # Site 5: Another config
    ]
    default_azimuths = [0, 120, 240]

    # Assign azimuths for each site before plotting
    for idx, site_id in enumerate(site_ids):
        site_mask = modified_site_config['site_id_temp'] == site_id
        site_cells = modified_site_config[site_mask]
        azimuths = azimuth_configs[idx] if idx < len(azimuth_configs) else default_azimuths
        logger.info(f"Site {site_id}: Using azimuths {azimuths}")
        for i, (row_idx, cell) in enumerate(site_cells.iterrows()):
            if i < len(azimuths):
                modified_site_config.loc[row_idx, 'cell_az_deg'] = azimuths[i]
                logger.info(f"Modified {cell['cell_id']}: azimuth = {azimuths[i]}°")
    
    # Replace the configuration temporarily
    original_config = visualizer.site_config_df_base
    visualizer.site_config_df_base = modified_site_config
    
    # Generate test plot
    try:
        visualizer.generate_comparison_plots(
            day=day,
            tick=tick,
            output_dir=output_dir
        )
        logger.info("Different azimuth angles test completed successfully!")
    finally:
        # Restore original configuration
        visualizer.site_config_df_base = original_config


def test_six_antennas_per_site(visualizer, day, tick, output_dir):
    """Test the visualizer with 6 antennas per site instead of 3."""
    logger.info("=== TESTING 6 ANTENNAS PER SITE ===")
    
    # Create shallow copy to avoid modifying the original data
    modified_site_config = visualizer.site_config_df_base.copy()
    
    # Create site_id_temp column if it doesn't exist
    if 'site_id_temp' not in modified_site_config.columns:
        modified_site_config['site_id_temp'] = modified_site_config['cell_id'].str.rsplit('_', n=1).str[0]
    
    # Get unique sites
    site_ids = modified_site_config['site_id_temp'].unique()
    
    # Create new rows for additional antennas (6 total per site)
    new_rows = []
    
    for site_id in site_ids:
        site_mask = modified_site_config['site_id_temp'] == site_id
        site_cells = modified_site_config[site_mask]
        
        # Get the first cell as template for this site
        template_cell = site_cells.iloc[0]
        
        # Define 6 azimuth angles (60° apart)
        azimuths = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
        
        # Update existing cells
        for i, (idx, cell) in enumerate(site_cells.iterrows()):
            if i < len(azimuths):
                modified_site_config.loc[idx, 'cell_az_deg'] = azimuths[i]
                logger.info(f"Modified {cell['cell_id']}: azimuth = {azimuths[i]}°")
        
        # Create additional cells for antennas 4, 5, 6
        for i in range(3, 6):  # antennas 4, 5, 6
            new_cell = template_cell.copy()
            new_cell['cell_id'] = f"{site_id}_cell_{i}"
            new_cell['cell_az_deg'] = azimuths[i]
            new_cell['site_id_temp'] = site_id
            
            # Update other fields to make it unique
            new_cell['ecgi'] = template_cell['ecgi'] + i * 1000  # Make ECGI unique
            new_cell['cell_name'] = f"{site_id}_Cell{i+1}"
            new_cell['enodeb_id'] = template_cell['enodeb_id']
            new_cell['tac'] = template_cell['tac']
            new_cell['cell_lat'] = template_cell['cell_lat']
            new_cell['cell_lon'] = template_cell['cell_lon']
            new_cell['cell_carrier_freq_mhz'] = template_cell['cell_carrier_freq_mhz']
            new_cell['cell_el_deg'] = template_cell['cell_el_deg']
            
            new_rows.append(new_cell)
            logger.info(f"Created new cell {new_cell['cell_id']}: azimuth = {azimuths[i]}°")
    
    # Add new rows to the configuration
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        modified_site_config = pd.concat([modified_site_config, new_df], ignore_index=True)
    
    # Replace the configuration temporarily
    original_config = visualizer.site_config_df_base
    visualizer.site_config_df_base = modified_site_config
    
    # Generate test plot
    try:
        visualizer.generate_comparison_plots(
            day=day,
            tick=tick,
            output_dir=output_dir
        )
        logger.info(f"6 antennas per site test completed successfully!")
        logger.info(f"Total cells: {len(modified_site_config)}, Cells per site: {len(modified_site_config) // len(site_ids)}")
    finally:
        # Restore original configuration
        visualizer.site_config_df_base = original_config


def test_visualization():
    """Test the visualization with azimuth-aligned wedges."""
    
    # Set up paths relative to project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
    BASE_DATA_DIR = os.path.join(NOTEBOOKS_DIR, "data", "energy_saving_data", "generated_data")
    STATIC_DATA_DIR = os.path.join(NOTEBOOKS_DIR, "data", "energy_saving_data", "data")
    
    # Model and data paths (using the already-trained models)
    BDT_MODEL_PATH = os.path.join(NOTEBOOKS_DIR, "data", "energy_saving_data", "bdt_model_map.pickle")
    RL_MODEL_PATH = os.path.join(NOTEBOOKS_DIR, "data", "energy_saving_data", "energy_saver_agent.zip")
    TOPOLOGY_PATH = os.path.join(STATIC_DATA_DIR, "topology.csv")
    CONFIG_PATH = os.path.join(STATIC_DATA_DIR, "config.csv")
    
    # Test output directory (local to the test directory)
    TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
    
    # Test parameters
    TEST_DAY = 1
    TICK = 16
    
    try:
        logger.info("Testing visualization with azimuth-aligned wedges...")
        logger.info(f"Using models from: {BDT_MODEL_PATH} and {RL_MODEL_PATH}")
        
        # Create visualizer
        visualizer = EnergySavingVisualizer(
            bdt_model_path=BDT_MODEL_PATH,
            rl_model_path=RL_MODEL_PATH,
            topology_path=TOPOLOGY_PATH,
            config_path=CONFIG_PATH,
            base_ue_data_dir=BASE_DATA_DIR
        )
        
        # Run different test scenarios
        logger.info("Running test scenarios...")
        
        # Test 1: Basic azimuth angle demonstration
        logger.info("=== TEST 1: AZIMUTH ANGLE DEMONSTRATION ===")
        test_azimuth_angles()
        
        # Test 2: Different azimuth angles (non-evenly distributed)
        logger.info("=== TEST 2: DIFFERENT AZIMUTH ANGLES ===")
        test_different_azimuth_angles(visualizer, TEST_DAY, TICK, TEST_OUTPUT_DIR)
        
        # Test 3: 6 antennas per site
        #logger.info("=== TEST 3: 6 ANTENNAS PER SITE ===")
        #test_six_antennas_per_site(visualizer, TEST_DAY, TICK, TEST_OUTPUT_DIR)
        
        logger.info("Visualization test completed successfully!")
        logger.info(f"Check the output at: {TEST_OUTPUT_DIR}")
        
    except Exception as e:
        logger.exception(f"Visualization test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_visualization()
    sys.exit(0 if success else 1) 