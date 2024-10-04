import streamlit as st
from EDA import EDA

file_path = st.file_uploader("Upload CSV File", type=['csv'])
if file_path is not None:
    eda = EDA(file_path)
    st.write("## Data Overview")
    eda.overview_of_data()
    st.write("## Summary Statistics")
    eda.summary_statistics()
    st.write("## Distribution of Numerical Features")
    eda.distribution_of_numerical_features()
    st.write("## Distribution of Categorical Features")
    eda.distribution_of_categorical_features()
    st.write("## Correlation Analysis")
    eda.correlation_analysis()
    st.write("## Missing Values")
    eda.identify_missing_values()
    st.write("## Outlier Detection")
    eda.detect_outliers()