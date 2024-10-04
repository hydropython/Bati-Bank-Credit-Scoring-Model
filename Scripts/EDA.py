import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
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
        numerical_cols = ['CountryCode', 'Amount', 'Value', 'PricingStrategy', 'FraudResult']
        
        # Create subplots with 2 rows and 3 columns
        num_plots = len(numerical_cols)
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), squeeze=False)  # Adjust size as needed
        
        # Flatten the axs array for easy iteration
        axs = axs.flatten()
        
        for ax, col in zip(axs, numerical_cols):
            if col == 'CountryCode':
                # For categorical-like numerical features, you can use a count plot
                sns.countplot(data=self.data, x=col, palette='pastel', ax=ax)
                ax.set_title(f'Distribution of {col}', fontsize=16)
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.tick_params(axis='x', rotation=45)
            else:
                # For other numerical features, use a histogram
                sns.histplot(data=self.data, x=col, bins=30, kde=True, ax=ax, color='skyblue')
                ax.set_title(f'Distribution of {col}', fontsize=16)
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)

            ax.grid(True)  # Add gridlines

        # Hide any unused subplots if there are less than 6 plots
        for i in range(num_plots, len(axs)):
            fig.delaxes(axs.flatten()[i])
        
        plt.tight_layout()
        plt.savefig("../Images/distribution_of_numerical_features.png")  # Save all subplots as one image
        plt.show()

    def distribution_of_categorical_features(self):
        meaningful_categorical_cols = ['CurrencyCode', 'ProductCategory', 'ChannelId']
        
        # Create subplots
        fig, axs = plt.subplots(nrows=1, ncols=len(meaningful_categorical_cols), figsize=(18, 6))
        
        for ax, col in zip(axs, meaningful_categorical_cols):
            sns.countplot(data=self.data, x=col, palette='pastel', ax=ax)
            ax.set_title(f'Distribution of {col}', fontsize=16)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True)  # Add gridlines
        
        plt.tight_layout()
        plt.savefig("../Images/distribution_of_categorical_features.png")  # Save all subplots as one image
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
            
            # Create subplots for box plots
            num_plots = len(numerical_cols)
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), squeeze=False)  # Adjust size as needed
            axs = axs.flatten()  # Flatten the array for easy iteration

            for idx, col in enumerate(numerical_cols):
                if method == 'IQR':
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Detect outliers
                    outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                    outliers_dict[col] = outliers
                    
                    # Calculate percentage of outliers
                    total_points = len(self.data[col])
                    num_outliers = len(outliers)
                    percent_outliers = (num_outliers / total_points) * 100
                    
                    print(f"{num_outliers} outliers detected in {col} using IQR method. ({percent_outliers:.2f}% of data)")
                
                # Create a box plot for the current column
                sns.boxplot(data=self.data, y=col, ax=axs[idx])
                axs[idx].set_title(f'Box Plot of {col}', fontsize=16)
                axs[idx].set_ylabel(col, fontsize=12)
                axs[idx].grid(True)  # Add gridlines

            # Hide any unused subplots if there are less than 6 plots
            for i in range(num_plots, len(axs)):
                fig.delaxes(axs.flatten()[i])
            
            plt.tight_layout()
            plt.savefig("../Images/outlier_detection_box_plots.png")  # Save all subplots as one image
            plt.show()
            
            return outliers_dict

   

    def handle_outliers(self, save_path="../Data/clean_data.csv"):
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numerical_cols:
            if col == 'Amount':
                # Log transformation for Amount to reduce skew
                print(f"Applying log transformation to {col}")
                self.data[col] = np.log1p(self.data[col])  # log1p to handle zeros as well
                
            elif col == 'Value':
                # Winsorization for Value (cap extreme values)
                print(f"Applying Winsorization to {col}")
                lower_percentile = self.data[col].quantile(0.05)
                upper_percentile = self.data[col].quantile(0.95)
                self.data[col] = np.clip(self.data[col], lower_percentile, upper_percentile)
            
            elif col == 'PricingStrategy':
                # Binning PricingStrategy to group outliers into categories
                print(f"Applying binning to {col}")
                self.data[col] = pd.cut(self.data[col], bins=[-np.inf, 1, 2, np.inf], labels=['Low', 'Medium', 'High'])
            
            elif col == 'FraudResult':
                # No outlier removal for FraudResult, retain as-is
                print(f"Retaining outliers in {col}, as they are crucial for analysis")
            
            else:
                # Default method if needed for other columns
                print(f"Handling outliers for {col} using default method (Winsorization)")
                lower_percentile = self.data[col].quantile(0.05)
                upper_percentile = self.data[col].quantile(0.95)
                self.data[col] = np.clip(self.data[col], lower_percentile, upper_percentile)

        # Save the cleaned data to a CSV file
        self.data.to_csv(save_path, index=False)
        print(f"Cleaned data saved to {save_path}")
        
        return self.data