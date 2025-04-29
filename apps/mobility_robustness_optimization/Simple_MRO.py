from .mobility_robustness_optimization import MobilityRobustnessOptimization
from notebooks.MRO_library import *
from radp.digital_twin.utils.constants import RLF_THRESHOLD

class SimpleMRO(MobilityRobustnessOptimization):
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
        Solve the mobility robustness optimization problem.
        """
        epochs = 100

        hyst = 0.00
        ttt = 2
        rlf_threshold = RLF_THRESHOLD

        attached_df = perform_attachment_hyst_ttt(self.simulation_data, hyst, ttt, rlf_threshold)

        max_diff = find_hyst_diff(attached_df)
        num_ticks = self.simulation_data["tick"].nunique()

        hyst_range = [0, max_diff]
        ttt_range = [2, num_ticks+1]
        score = pd.DataFrame(columns=["hyst", "ttt", "score"])

        score.loc[len(score)] = [hyst, ttt, calculate_mro_metric(attached_df)]
        for i in range(epochs):
            while True:
                hyst = np.random.uniform(hyst_range[0], hyst_range[1])
                ttt = np.random.randint(ttt_range[0], ttt_range[1])
                if ttt not in score["ttt"].values or hyst not in score["hyst"].values:
                    break
            # Perform attachment and calculate MRO Metric
            attached_df = perform_attachment_hyst_ttt(self.simulation_data, hyst, ttt, rlf_threshold)
            mro_metric = calculate_mro_metric(attached_df)

            # Store the data in the score DataFrame
            score.loc[len(score)] = [hyst, ttt, mro_metric]
        
        return (score.loc[score["score"].idxmax(), "hyst"], int(score.loc[score["score"].idxmax(), "ttt"]))

        