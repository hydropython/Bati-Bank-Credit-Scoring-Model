import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

class RFMSProcessor:
    def __init__(self, df):
        self.df = df
        self.rfms_df = None

    def calculate_rfms(self):
        # Convert 'TransactionStartTime' to datetime
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'], errors='coerce')

        # Check if 'TransactionStartTime' is timezone naive or aware
        if self.df['TransactionStartTime'].dt.tz is None:
            # Localize to UTC if it's naive
            self.df['TransactionStartTime'] = self.df['TransactionStartTime'].dt.tz_localize('UTC')
        else:
            # Convert to UTC if it's already aware
            self.df['TransactionStartTime'] = self.df['TransactionStartTime'].dt.tz_convert('UTC')

        # Calculate Recency, Frequency, and Monetary metrics
        current_date = pd.to_datetime("now", utc=True)  # Make current_date timezone-aware

        self.rfms_df = self.df.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda x: (current_date - x.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Amount', 'sum')
        ).reset_index()

        # Calculate RFMS Score (you can adjust the weights as needed)
        # For example, we will create a score where lower recency is better (hence multiplied by -1)
        # Adjust monetary value scale based on business logic
        self.rfms_df['Score'] = (self.rfms_df['Recency'] * -1 + 
                                self.rfms_df['Frequency'] + 
                                self.rfms_df['Monetary'] / 1000)  # Adjust the scale as necessary

        # Check if RFMS DataFrame has been created successfully
        print("RFMS DataFrame:\n", self.rfms_df.head())
        
    def visualize_rfms(self):
        """Visualize RFMS scores using Matplotlib with subplots."""
        if self.rfms_df is None:
            raise ValueError("RFMS scores have not been calculated. Please run calculate_rfms() first.")

        if 'Score' not in self.rfms_df.columns:
            raise ValueError("Score column is missing from the RFMS DataFrame. Please check the calculations.")

        # Set the style
        sns.set(style='whitegrid')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('RFMS Analysis', fontsize=20)

        # Recency Plot
        sns.histplot(self.rfms_df['Recency'], bins=30, ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Recency Distribution')
        axes[0, 0].set_xlabel('Days Since Last Purchase')
        axes[0, 0].set_ylabel('Number of Customers')
        axes[0, 0].grid(linestyle='-', linewidth=0.5)  # Lighter grid

        # Frequency Plot
        sns.histplot(self.rfms_df['Frequency'], bins=30, ax=axes[0, 1], color='lightgreen')
        axes[0, 1].set_title('Frequency Distribution')
        axes[0, 1].set_xlabel('Number of Transactions')
        axes[0, 1].set_ylabel('Number of Customers')
        axes[0, 1].grid(linestyle='-', linewidth=0.5)  # Lighter grid

        # Monetary Plot
        sns.histplot(self.rfms_df['Monetary'], bins=30, ax=axes[1, 0], color='salmon')
        axes[1, 0].set_title('Monetary Distribution')
        axes[1, 0].set_xlabel('Total Amount Spent')
        axes[1, 0].set_ylabel('Number of Customers')
        axes[1, 0].grid(linestyle='-', linewidth=0.5)  # Lighter grid

        # Score Scatter Plot
        sns.scatterplot(x=self.rfms_df.index, y=self.rfms_df['Score'], ax=axes[1, 1], color='gold', s=100, edgecolor='w')
        axes[1, 1].set_title('RFMS Scores by Customer')
        axes[1, 1].set_xlabel('Customer Index')
        axes[1, 1].set_ylabel('RFMS Score')
        axes[1, 1].grid(linestyle='-', linewidth=0.5)  # Lighter grid

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    def classify_users(self, monetary_threshold=1000, recency_threshold=30):
        """Classify users as 'Good' or 'Bad' based on Recency and Monetary metrics."""
        conditions = [
            (self.rfms_df['Recency'] <= recency_threshold) & (self.rfms_df['Monetary'] >= monetary_threshold),  # Good
            (self.rfms_df['Recency'] > recency_threshold) & (self.rfms_df['Monetary'] < monetary_threshold),  # Bad
        ]
        
        labels = ['Good', 'Bad']
        
        # Assign labels based on conditions
        self.rfms_df['UserLabel'] = np.select(conditions, labels, default='Neutral')  # Default can be 'Neutral'
        
        print("User labeling completed.")
        print(self.rfms_df[['CustomerId', 'Recency', 'Monetary', 'UserLabel']].head())
                        
    def visualize_user_labels(self):
        """Visualize the distribution of user labels."""
        if 'UserLabel' not in self.rfms_df.columns:
            raise ValueError("User labeling has not been done. Please run classify_users() first.")

        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.rfms_df, x='UserLabel', palette='pastel')
        plt.title('Distribution of User Labels', fontsize=16)
        plt.xlabel('User Label', fontsize=14)
        plt.ylabel('Number of Users', fontsize=14)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.show()