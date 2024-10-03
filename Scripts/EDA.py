import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

class EDA:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        # Ensure the Images directory exists
        os.makedirs('../Images', exist_ok=True)

    def overview_of_data(self):
        # Overview of the dataset
        print(f"### Overview of Data ###")
        print(f"Number of Rows: {self.data.shape[0]}")
        print(f"Number of Columns: {self.data.shape[1]}")
        print(self.data.info())

    def summary_statistics(self):
        # Summary statistics for numerical data
        print("### Summary Statistics for Numerical Data ###")
        num_desc = self.data.describe(include=['float64', 'int64'])
        print(num_desc)

        # Summary statistics for categorical data
        print("\n### Summary Statistics for Categorical Data ###")
        cat_desc = self.data.describe(include=['object'])
        print(cat_desc)

    def distribution_of_numerical_features(self):
        # Visualizing the distribution of numerical features
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        n = len(numerical_cols)

        # Calculate the number of rows needed for the subplots
        rows = (n + 1) // 2

        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(12, 5 * rows))
        fig.suptitle('Distribution of Numerical Features', fontsize=16)

        for i, col in enumerate(numerical_cols):
            row, col_idx = divmod(i, 2)
            sns.histplot(self.data[col], kde=True, ax=axes[row, col_idx], color='teal', bins=30)
            axes[row, col_idx].set_title(f'Distribution of {col}', fontsize=14)
            axes[row, col_idx].set_xlabel(col, fontsize=12)
            axes[row, col_idx].set_ylabel('Frequency', fontsize=12)
            axes[row, col_idx].grid(True)

        # Remove any empty subplots if n is odd
        if n % 2 != 0:
            fig.delaxes(axes[rows - 1, 1])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('../Images/distribution_of_numerical_features.png')
        plt.show()  # Show the plot in the notebook
        plt.close(fig)

    def distribution_of_categorical_features(self):
        # Visualizing the distribution of categorical features
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        n = len(categorical_cols)

        # Calculate the number of rows needed for the subplots
        rows = (n + 1) // 2

        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(12, 5 * rows))
        fig.suptitle('Distribution of Categorical Features', fontsize=16)

        for i, col in enumerate(categorical_cols):
            row, col_idx = divmod(i, 2)
            sns.countplot(x=col, data=self.data, ax=axes[row, col_idx], palette='viridis')
            axes[row, col_idx].set_title(f'Distribution of {col}', fontsize=14)
            axes[row, col_idx].set_xlabel(col, fontsize=12)
            axes[row, col_idx].set_ylabel('Count', fontsize=12)
            axes[row, col_idx].grid(True)

        # Remove any empty subplots if n is odd
        if n % 2 != 0:
            fig.delaxes(axes[rows - 1, 1])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('../Images/distribution_of_categorical_features.png')
        plt.show()  # Show the plot in the notebook
        plt.close(fig)

    def correlation_analysis(self):
        # Correlation analysis between numerical features
        correlation_matrix = self.data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Heatmap', fontsize=16)
        plt.grid(True)
        plt.savefig('../Images/correlation_heatmap.png')
        plt.show()  # Show the plot in the notebook
        plt.close()

    def identify_missing_values(self):
        # Identify missing values
        missing_values = self.data.isnull().sum()
        missing_values = missing_values[missing_values > 0]

        if len(missing_values) > 0:
            print("\n### Missing Values by Feature ###")
            print(missing_values)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=missing_values.index, y=missing_values.values, palette='Set2')
            plt.title('Missing Values by Feature', fontsize=16)
            plt.ylabel('Number of Missing Values', fontsize=12)
            plt.xlabel('Features', fontsize=12)
            plt.grid(True)
            plt.savefig('../Images/missing_values.png')
            plt.show()  # Show the plot in the notebook
            plt.close()
        else:
            print("No missing values in the dataset.")

    def outlier_detection(self):
        # Use box plots to identify outliers
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        n = len(numerical_cols)

        rows = (n + 1) // 2

        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(12, 5 * rows))
        fig.suptitle('Outlier Detection with Box Plots', fontsize=16)

        for i, col in enumerate(numerical_cols):
            row, col_idx = divmod(i, 2)
            sns.boxplot(y=self.data[col], ax=axes[row, col_idx], palette='pastel')
            axes[row, col_idx].set_title(f'Box Plot of {col}', fontsize=14)
            axes[row, col_idx].grid(True)

        if n % 2 != 0:
            fig.delaxes(axes[rows - 1, 1])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('../Images/outlier_detection.png')
        plt.show()  # Show the plot in the notebook
        plt.close(fig)