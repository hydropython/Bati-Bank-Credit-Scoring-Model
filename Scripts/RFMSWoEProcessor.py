import pandas as pd
import numpy as np
import os

class RFMSWoEProcessor:
    def __init__(self, data):
        self.data = data

    def calculate_woe(self, feature, target, bins=None):
        """
        Calculate the Weight of Evidence (WoE) for a given feature against a target variable.
        
        Parameters:
        - feature (str): The name of the feature to calculate WoE for.
        - target (str): The name of the binary target variable.
        - bins (int or list, optional): Number of bins or specific bin edges to use. 
                                         If None, quantile bins will be created.

        Returns:
        - pd.DataFrame: DataFrame containing binned feature, total counts, good/bad counts, 
                        distributions, and WoE values.
        """
        # Bin the feature
        if bins is None:
            self.data[feature + '_bin'] = pd.qcut(self.data[feature], q=10, duplicates='drop')
        else:
            self.data[feature + '_bin'] = pd.cut(self.data[feature], bins=bins)

        # Group by the binned feature and calculate good/bad counts
        grouped = self.data.groupby(feature + '_bin', observed=False)[target].agg(
            total='count', 
            good='sum', 
            bad=lambda x: x.count() - x.sum()
        ).reset_index()

        # Calculate distributions
        grouped['good_dist'] = grouped['good'] / grouped['good'].sum() if grouped['good'].sum() > 0 else 0
        grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum() if grouped['bad'].sum() > 0 else 0

        # Calculate WoE, handling division by zero
        grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist']).replace([np.inf, -np.inf], 0)

        return grouped

    def run_woe(self):
        """
        Run WoE transformation on specified features and save the results.
        """
        # Apply WoE transformation to Recency
        print("Recency WoE Table:")
        recency_woe = self.calculate_woe('Recency', 'Score')  # 'Score' is the binary target
        print(recency_woe)

        # Apply WoE transformation to Monetary
        print("Monetary WoE Table:")
        monetary_woe = self.calculate_woe('Monetary', 'Score')  # 'Score' is the binary target
        print(monetary_woe)

        # Save the outputs
        self.save_woe_output(recency_woe, "../Data/recency_woe_output.csv")
        self.save_woe_output(monetary_woe, "../Data/monetary_woe_output.csv")

    def save_woe_output(self, woe_df, file_name):
        """
        Save the WoE DataFrame to a CSV file.
        
        Parameters:
        - woe_df (pd.DataFrame): The DataFrame containing WoE calculations.
        - file_name (str): The name of the file to save the DataFrame to.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        
        woe_df.to_csv(file_name, index=False)
        print(f"WoE output saved to {file_name}")