import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

class EDA:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        os.makedirs('../Images', exist_ok=True)
        os.makedirs('../Data', exist_ok=True)

    def overview_of_data(self):
        print(f"### Overview of Data ###")
        print(f"Number of Rows: {self.data.shape[0]}")
        print(f"Number of Columns: {self.data.shape[1]}")
        print(f"\nData Types:\n{self.data.dtypes}")
        print(f"\nMissing Values:\n{self.data.isnull().sum()}")
        
        return self.data.shape, self.data.dtypes, self.data.isnull().sum()

    def summary_statistics(self):
        numerical_summary = self.data.describe(include=['float64', 'int64'])
        print("### Summary Statistics for Numerical Data ###")
        print(numerical_summary)

        categorical_summary = self.data.describe(include=['object'])
        print("\n### Summary Statistics for Categorical Data ###")
        print(categorical_summary)

        return numerical_summary, categorical_summary

    def distribution_of_numerical_features(self):
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        print(f"Numerical Columns: {numerical_cols}")
        
        n = len(numerical_cols)
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

        if n % 2 != 0:
            fig.delaxes(axes[rows - 1, 1])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('../Images/distribution_of_numerical_features.png')
        plt.show()
        plt.close(fig)

    def distribution_of_categorical_features(self):
        meaningful_categorical_cols = ['CurrencyCode', 'ProductCategory', 'ChannelId']
        
        for col in meaningful_categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.data, x=col, palette='pastel')
            plt.title(f'Distribution of {col}', fontsize=16)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"../Images/distribution_of_{col}.png")
            plt.show()

    def correlation_analysis(self):
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        
        correlation_matrix = numeric_data.corr()
        
        means = numeric_data.mean()
        stds = numeric_data.std()
        counts = numeric_data.count()

        print("### Correlation Analysis ###")
        print("\nCorrelation Matrix:")
        print(correlation_matrix)
        
        print("\nMeans:")
        print(means)
        
        print("\nStandard Deviations:")
        print(stds)
        
        print("\nCounts of Non-NA Values:")
        print(counts)

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    fmt=".2f", 
                    cmap='Blues', 
                    square=True, 
                    cbar_kws={"shrink": .8}, 
                    linewidths=.5, 
                    linecolor='black')
        
        plt.title('Correlation Heatmap', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig('../Images/correlation_heatmap_matplotlib.png')
        plt.show()
        
        return correlation_matrix, means, stds, counts

    def identify_missing_values(self):
        missing_values = self.data.isnull().sum()
        missing_values = missing_values[missing_values > 0]

        if len(missing_values) > 0:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=missing_values.index, y=missing_values.values, palette='pastel')
            plt.title('Missing Values Count per Column', fontsize=16)
            plt.xlabel('Columns', fontsize=12)
            plt.ylabel('Count of Missing Values', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('../Images/missing_values_count.png')
            plt.show()
        else:
            print("No missing values in the dataset.")

    def detect_outliers(self, method='IQR'):
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        outliers_dict = {}

        for col in numerical_cols:
            if method == 'IQR':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                outliers_dict[col] = outliers
                print(f"{len(outliers)} outliers detected in {col} using IQR method.")
        
        return outliers_dict

    def handle_outliers(self, method='remove'):
        if method not in ['remove', 'replace']:
            raise ValueError("Method must be 'remove' or 'replace'.")

        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numerical_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'remove':
                initial_count = self.data.shape[0]
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
                final_count = self.data.shape[0]
                print(f"Removed {initial_count - final_count} outliers from {col}.")
            elif method == 'replace':
                self.data[col] = self.data[col].mask(self.data[col] < lower_bound, lower_bound)
                self.data[col] = self.data[col].mask(self.data[col] > upper_bound, upper_bound)
                print(f"Replaced outliers in {col} with bounds.")