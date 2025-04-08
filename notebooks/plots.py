import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import seaborn as sns

def plot_3d_hyst_ttt_score(df):
    """
    Creates an interactive 3D scatter plot of hyst, ttt, and score.

    Parameters:
        df (pd.DataFrame): A DataFrame containing columns 'hyst', 'ttt', and 'score'.
    """
    fig = px.scatter_3d(df, x='hyst', y='ttt', z='score',
                        color='score', color_continuous_scale='viridis',
                        title='3D Scatter Plot of Hyst, TTT, and Score')
    fig.update_traces(marker=dict(size=6))
    fig.show()


def plot_scatter2(df, topology):
    # Create a figure and axis
    plt.figure(figsize=(10, 8))

    plt.scatter([], [], color='grey', label='RLF')

    # Define color mapping based on cell_id for both cells and UEs
    color_map = {1.0: 'red', 2.0: 'green', 3.0: 'blue'}

    # Plot cell towers from the topology dataframe with 'X' markers and corresponding colors
    for _, row in topology.iterrows():
        color = color_map.get(float(row['cell_id']), 'black')  # Ensure it's a float to match df cell_id format
        plt.scatter(row['cell_lon'], row['cell_lat'], marker='X', s=200, linewidths=2, c=[color], label=f"Cell {row['cell_id']}")

    # Plot UEs from df without labels but with the same color coding
    for _, row in df.iterrows():
        cell_id = row['cell_id']
        if cell_id == "RLF":
            color = 'grey'  # If cell_id is "RLF", set color to grey
        else:
            color = color_map.get(cell_id, 'black')  # Get color from the color_map

        plt.scatter(row['loc_x'], row['loc_y'], c=[color])

    # Add labels and title
    plt.xlabel('Longitude (loc_x)')
    plt.ylabel('Latitude (loc_y)')
    plt.title('Cell Towers and UE Locations')

    # Create a legend for the cells only
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def plot_ue_rxpower_over_time(df: pd.DataFrame, total_ue: pd.DataFrame, ue_id: int):
    """
    Plots cell_rxpower_dbm over tick for a specific mock_ue_id with line color changing based on cell_id.
    Also overlays the data for the given ue_id using a green dotted line, ensuring that when cell_id is "RLF",
    the dotted line is at the bottom of the graph.
    """
    # Filter DataFrames
    df_filtered = df[df['ue_id'] == ue_id].sort_values(by='tick')
    total_filtered = total_ue[total_ue['ue_id'] == ue_id].sort_values(by='tick')

    if df_filtered.empty or total_filtered.empty:
        print(f"No data found for ue_id {ue_id}.")
        return

    # Determine minimum power level for plotting RLF
    min_power = total_filtered['cell_rxpower_dbm'].min()

    # Assign colors for total_ue dataset
    unique_cells_total = total_filtered['cell_id'].unique()
    colors_total = sns.color_palette("husl", len(unique_cells_total))
    cell_color_map_total = {cell: colors_total[i] for i, cell in enumerate(unique_cells_total)}

    # Plot the total_ue dataset
    plt.figure(figsize=(12, 6))
    for cell in unique_cells_total:
        cell_data = total_filtered[total_filtered['cell_id'] == cell]
        plt.plot(cell_data['tick'], cell_data['cell_rxpower_dbm'], label=f"Cell {cell}",
                 color=cell_color_map_total[cell], linewidth=2)

    # Overlay the selected df with a green dotted line
    rlf_ticks = []
    rlf_values = []
    normal_ticks = []
    normal_values = []

    for _, row in df_filtered.iterrows():
        if row['cell_id'] == "RLF":
            rlf_ticks.append(row['tick'])
            rlf_values.append(min_power)
        else:
            normal_ticks.append(row['tick'])
            normal_values.append(row['cell_rxpower_dbm'])

    if rlf_ticks:
        plt.plot(rlf_ticks, rlf_values, 'k--', linewidth=2, label="RLF")  # Dotted line at bottom
    if normal_ticks:
        plt.plot(normal_ticks, normal_values, 'k--', linewidth=2, label="Connected To Cell")

    # Labels and title
    plt.xlabel("Tick (Time)")
    plt.ylabel("Cell RX Power (dBm)")
    plt.title(f"Cell RX Power over Time for UE: {ue_id}")

    # Legend
    plt.legend(title="Cell ID")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def individual_scatter_plot(df, topology, ue_id):
    # Create a figure and axis
    plt.figure(figsize=(10, 8))

    plt.scatter([], [], color='grey', label='RLF')

    # Define color mapping based on cell_id for both cells and UEs
    color_map = {1.0: 'red', 2.0: 'green', 3.0: 'blue'}

    # Plot cell towers from the topology dataframe with 'X' markers and corresponding colors
    for _, row in topology.iterrows():
        color = color_map.get(row['cell_id'], 'black')  # Default to black if unknown cell_id
        plt.scatter(row['cell_lon'], row['cell_lat'], marker='X', s=200, linewidths=2, c=[color], label=f"Cell {row['cell_id']}")

    # Filter df to only include the selected ue_id
    df_filtered = df[df['ue_id'] == ue_id]

    # Plot only the selected UE
    for _, row in df_filtered.iterrows():
        cell_id = row['cell_id']
        if cell_id == "RLF":
            color = 'grey'  # If cell_id is "RLF", set color to grey
        else:
            cell_id_str = f"cell_{int(cell_id)}"  # Convert float cell_id to corresponding string format
            color = color_map.get(cell_id_str, 'black')  # Default to black if unknown cell_id

        plt.scatter(row['loc_x'], row['loc_y'], c=[color], label=f"UE {row['ue_id']}")

    # Add labels and title
    plt.xlabel('Longitude (loc_x)')
    plt.ylabel('Latitude (loc_y)')
    plt.title(f'Cell Towers and UE {ue_id} Location')

    # Create a legend for the cells only
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Show the plot
    plt.show()
