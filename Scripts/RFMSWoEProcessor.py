import pandas as pd
import numpy as np
import os

class RFMSWoEProcessor:
    def __init__(self, data):
        self.data = data

    def calculate_woe(self, feature, target, bins=None):
        # Bin the feature
        bin_col_name = feature + '_bin'
        if bins is None:
            self.data[bin_col_name] = pd.qcut(self.data[feature], q=10, duplicates='drop')
        else:
            self.data[bin_col_name] = pd.cut(self.data[feature], bins=bins)

        # Group by the binned feature and calculate good/bad counts
        grouped = self.data.groupby(bin_col_name)[target].agg(
            total='count', 
            good='sum', 
            bad=lambda x: x.count() - x.sum()
        ).reset_index()

        # Calculate distributions
        if grouped['good'].sum() > 0 and grouped['bad'].sum() > 0:
            grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
            grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
        else:
            grouped['good_dist'] = 0
            grouped['bad_dist'] = 0

        # Calculate WoE, handling division by zero
        grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist']).replace([np.inf, -np.inf], 0)

        return grouped

    def run_woe_per_customer(self, customer_id):
        """
        Run WoE calculation for each unique CustomerId and save results to CSV.
        
        Parameters:
        - customer_id (str): The column name for CustomerId.
        """
        # Group by CustomerId
        unique_customers = self.data[customer_id].unique()
        all_woe_results = {}

        for customer in unique_customers:
            customer_data = self.data[self.data[customer_id] == customer]
            
            # Calculate WoE for Recency
            recency_woe = self.calculate_woe('Recency', 'Score')
            all_woe_results[customer] = {
                'Recency': recency_woe
            }

            # Calculate WoE for Monetary
            monetary_woe = self.calculate_woe('Monetary', 'Score')
            all_woe_results[customer]['Monetary'] = monetary_woe

        # Save results to CSV
        results_df = pd.DataFrame()
        for customer, woe_data in all_woe_results.items():
            # Get the dynamically created bin column names
            recency_bin_col = 'Recency_bin'
            monetary_bin_col = 'Monetary_bin'
            
            # Assign CustomerId and rename bin columns for consistent merging
            recency_df = woe_data['Recency'].rename(columns={recency_bin_col: 'Bin'}).assign(CustomerId=customer)
            monetary_df = woe_data['Monetary'].rename(columns={monetary_bin_col: 'Bin'}).assign(CustomerId=customer)
            
            # Merge on the common 'Bin' and 'CustomerId' columns
            combined_df = recency_df.merge(monetary_df, on=['Bin', 'CustomerId'], how='outer', suffixes=('_recency', '_monetary'))
            results_df = pd.concat([results_df, combined_df], ignore_index=True)

        # Specify the file path
        output_file = '../Data/woe_results.csv'
        results_df.to_csv(output_file, index=False)

        return all_woe_results