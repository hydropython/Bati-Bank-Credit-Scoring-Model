
# Bati-Bank-Credit-Scoring-Model
This repository contains a Credit Scoring Model for Bati Bank's collaboration with an eCommerce company to enable a Buy-Now-Pay-Later service. The model assesses creditworthiness and predicts customer default risk, following Basel II Capital Accord guidelines to support informed credit decisions and minimize risk.

Repository Structure
---

Scripts:
---
The primary analysis and modeling steps are provided as Python scripts:

*eda.py: Script for performing exploratory data analysis on the dataset.

*feature_engineering.py: Script to engineer and transform features for better model performance.

*rfmsprocesser.py: Script for RFM (Recency, Frequency, Monetary) analysis and customer segmentation.

*rfmswoeprocesser.py: Script for applying Weight of Evidence (WOE) transformation to RFM segments.

*model.py: Script for building and evaluating machine learning models.

Notebooks
---

The notebook folder contains Jupyter notebooks that complement the scripts by providing:

*eda.ipynb: A more detailed, interactive version of the EDA process.

*feature_engineering.ipynb: Demonstrates feature engineering techniques interactively.

*rfmsprocesser.ipynb: Explores the RFM segmentation results in an interactive format.

*rfmswoeprocesser.ipynb: Shows the process of WOE encoding and Information Value calculations.

*model.ipynb: Walkthrough of the model development process, including model evaluation and hyperparameter tuning.


How to Use
---

Running the Python Scripts
1. Clone the repository:
git clone https://github.com/yourusername/your-repo-name.git


2. Install the required libraries:
pip install -r requirements.txt


3. Run each script in sequence as per the project requirement:
python eda.py
python feature_engineering.py
python rfmsprocesser.py
python rfmswoeprocesser.py
python model.py


Project Workflow
Exploratory Data Analysis (EDA):
The eda.py script explores the dataset to identify key patterns, outliers, and relationships between variables. This step includes data cleaning and visualization.

Feature Engineering:
In feature_engineering.py, new features are created, and existing features are transformed for better model performance. This step includes encoding categorical variables and handling missing data.

RFM Segmentation:
The rfmsprocesser.py script computes Recency, Frequency, and Monetary (RFM) values to segment customers. Insights from this segmentation are useful for targeted marketing strategies.

Weight of Evidence Transformation:
The rfmswoeprocesser.py applies Weight of Evidence (WOE) encoding to the RFM segments to prepare features for predictive modeling, improving model accuracy.

Model Development:
The model.py script builds machine learning models using techniques like logistic regression, decision trees, or other classifiers. The models are evaluated based on key performance metrics such as accuracy, precision, recall, and ROC-AUC.

Conclusion
---
This repository provides a complete framework for customer segmentation and predictive modeling using Python. Each step, from data analysis to feature engineering and modeling, is covered by well-structured scripts and notebooks. The project can be easily extended to handle additional datasets or customized for specific business needs.
