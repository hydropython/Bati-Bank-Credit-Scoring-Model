import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
import numpy as np

class EDA:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        os.makedirs('../Images', exist_ok=True)
        os.makedirs('../Data', exist_ok=True)

    def overview_of_data(self):
        st.write("### Overview of Data")
        st.write(f"Number of Rows: {self.data.shape[0]}")
        st.write(f"Number of Columns: {self.data.shape[1]}")
        
        st.write("### Data Types")
        st.write(self.data.dtypes)
        
        st.write("### Missing Values")
        st.write(self.data.isnull().sum())

        return self.data.shape, self.data.dtypes, self.data.isnull().sum()

    def summary_statistics(self):
        numerical_summary = self.data.describe(include=['float64', 'int64'])
        categorical_summary = self.data.describe(include=['object'])
        
        st.write("### Summary Statistics for Numerical Data")
        st.write(numerical_summary)
        
        st.write("### Summary Statistics for Categorical Data")
        st.write(categorical_summary)

        return numerical_summary, categorical_summary

    def distribution_of_numerical_features(self):
        numerical_cols = ['CountryCode', 'Amount', 'Value', 'PricingStrategy', 'FraudResult']
        
        st.write("### Distribution of Numerical Features")
        
        for col in numerical_cols:
            st.write(f"#### {col}")
            fig, ax = plt.subplots()
            if col == 'CountryCode':
                sns.countplot(data=self.data, x=col, palette='pastel', ax=ax)
            else:
                sns.histplot(data=self.data, x=col, bins=30, kde=True, color='skyblue', ax=ax)
            st.pyplot(fig)

    def distribution_of_categorical_features(self):
        meaningful_categorical_cols = ['CurrencyCode', 'ProductCategory', 'ChannelId']
        
        st.write("### Distribution of Categorical Features")
        
        for col in meaningful_categorical_cols:
            st.write(f"#### {col}")
            fig, ax = plt.subplots()
            sns.countplot(data=self.data, x=col, palette='pastel', ax=ax)
            st.pyplot(fig)

    def correlation_analysis(self):
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        
        correlation_matrix = numeric_data.corr()
        st.write("### Correlation Analysis")
        st.write("#### Correlation Matrix")
        st.write(correlation_matrix)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Blues', square=True, ax=ax)
        plt.title('Correlation Heatmap', fontsize=16)
        st.pyplot(fig)
        
        return correlation_matrix

    def identify_missing_values(self):
        missing_values = self.data.isnull().sum()
        missing_values = missing_values[missing_values > 0]

        st.write("### Missing Values")
        if len(missing_values) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=missing_values.index, y=missing_values.values, palette='pastel', ax=ax)
            st.pyplot(fig)
        else:
            st.write("No missing values in the dataset.")

    def detect_outliers(self, method='IQR'): 
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        outliers_dict = {}

        st.write("### Outlier Detection")
        
        for col in numerical_cols:
            if method == 'IQR':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                outliers_dict[col] = outliers

                num_outliers = len(outliers)
                percent_outliers = (num_outliers / len(self.data[col])) * 100
                st.write(f"{num_outliers} outliers detected in {col} using IQR method. ({percent_outliers:.2f}% of data)")
                
                fig, ax = plt.subplots()
                sns.boxplot(data=self.data, y=col, ax=ax)
                st.pyplot(fig)

        return outliers_dict

    def handle_outliers(self, save_path="../Data/clean_data.csv"):
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        st.write("### Outlier Handling")

        for col in numerical_cols:
            if col == 'Amount':
                st.write(f"Applying log transformation to {col}")
                self.data[col] = np.log1p(self.data[col])
            elif col == 'Value':
                st.write(f"Applying Winsorization to {col}")
                lower_percentile = self.data[col].quantile(0.05)
                upper_percentile = self.data[col].quantile(0.95)
                self.data[col] = np.clip(self.data[col], lower_percentile, upper_percentile)
            elif col == 'PricingStrategy':
                st.write(f"Applying binning to {col}")
                self.data[col] = pd.cut(self.data[col], bins=[-np.inf, 1, 2, np.inf], labels=['Low', 'Medium', 'High'])
            elif col == 'FraudResult':
                st.write(f"Retaining outliers in {col}, as they are crucial for analysis")
            else:
                st.write(f"Handling outliers for {col} using default Winsorization")
                lower_percentile = self.data[col].quantile(0.05)
                upper_percentile = self.data[col].quantile(0.95)
                self.data[col] = np.clip(self.data[col], lower_percentile, upper_percentile)

        self.data.to_csv(save_path, index=False)
        st.write(f"Cleaned data saved to {save_path}")

        return self.data