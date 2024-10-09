import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pandas.plotting import parallel_coordinates
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class ModelSelector:
    def __init__(self, data):
        self.data = data
        self.target_column = 'UserLabel'  # Set target variable
        self.models = {
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier()
        }
        self.results = {}

    def merge_data(self, df1, df2):
        """ Merge two dataframes on a common key, e.g., 'CustomerId' """
        self.data = pd.merge(df1, df2, on='CustomerId', how='inner')

    def preprocess_data(self):
        """ Preprocess the data by handling non-numeric columns and missing values """
        # Separate numeric and non-numeric columns
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns

        # Print to inspect column selection
        print("Numeric columns:", numeric_cols)
        print("Categorical columns:", categorical_cols)

        # Drop columns with all missing values
        self.data.dropna(axis=1, how='all', inplace=True)

        # Re-check numeric columns after dropping
        numeric_cols = self.data.select_dtypes(include=['number']).columns

        # Impute missing values for numeric columns with mean
        imputer_numeric = SimpleImputer(strategy='mean')
        imputed_numeric_data = imputer_numeric.fit_transform(self.data[numeric_cols])

        # Convert the imputed array back to a DataFrame
        self.data[numeric_cols] = pd.DataFrame(imputed_numeric_data, columns=numeric_cols)

        # Handle categorical columns: Label encode and impute with most frequent value
        for col in categorical_cols:
            # Apply Label Encoding for categorical variables
            encoder = LabelEncoder()
            self.data[col] = encoder.fit_transform(self.data[col].astype(str))

        # Impute missing values for categorical columns with the most frequent value
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        imputed_categorical_data = imputer_categorical.fit_transform(self.data[categorical_cols])
        # Convert the imputed array back to a DataFrame
        self.data[categorical_cols] = pd.DataFrame(imputed_categorical_data, columns=categorical_cols)

    def split_data(self, test_size=0.2, random_state=42):
        """ Split the data into training and testing sets """
        # Preprocess the data first
        self.preprocess_data()

        # Automatically selecting all columns except the target column
        self.feature_columns = [col for col in self.data.columns if col != self.target_column]

        X = self.data[self.feature_columns]
        y = self.data[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)

    def train_models(self):
        """ Train the models on the training data """
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
    
    def hyperparameter_tuning(self):
        """ Perform hyperparameter tuning for the models and store results. """
        hyperparameter_results = []

        # Random Forest hyperparameters
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search_rf = GridSearchCV(estimator=self.models['Random Forest'], param_grid=rf_param_grid, scoring='accuracy', cv=5)
        grid_search_rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = grid_search_rf.best_estimator_

        # Store Random Forest results
        for params, score in zip(grid_search_rf.cv_results_['params'], grid_search_rf.cv_results_['mean_test_score']):
            hyperparameter_results.append({**params, 'Model': 'Random Forest', 'Score': score})

        # Gradient Boosting hyperparameters
        gb_param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        grid_search_gb = GridSearchCV(estimator=self.models['Gradient Boosting'], param_grid=gb_param_grid, scoring='accuracy', cv=5)
        grid_search_gb.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = grid_search_gb.best_estimator_

        # Store Gradient Boosting results
        for params, score in zip(grid_search_gb.cv_results_['params'], grid_search_gb.cv_results_['mean_test_score']):
            hyperparameter_results.append({**params, 'Model': 'Gradient Boosting', 'Score': score})

        # Create DataFrame
        hyperparameter_df = pd.DataFrame(hyperparameter_results)

        # Reorder the columns to ensure 'Score' is at the end
        score_column = hyperparameter_df.pop('Score')
        hyperparameter_df['Score'] = score_column

        # Print the DataFrame
        print(hyperparameter_df)

        # Call plotting function
        self.plot_hyperparameter_results(hyperparameter_df)

    def plot_hyperparameter_results(self, hyperparameter_df):
        """ Create a parallel coordinates plot for hyperparameter tuning results """
        plt.figure(figsize=(12, 6))
        parallel_coordinates(hyperparameter_df, 'Model', color=['#1f77b4', '#ff7f0e'], linewidth=2)
        plt.title('Parallel Coordinates Plot of Hyperparameter Tuning')
        plt.xlabel('Hyperparameters and Score')
        plt.ylabel('Values')
        plt.grid(True)
        plt.legend(title='Model')
        plt.savefig('../Images/hyperparameter_tuning_plot.png')
        plt.show()
        
    def evaluate_models(self):
        """ Evaluate the models using various metrics """
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            self.results[model_name] = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred, pos_label='Good'),  # Specify pos_label
                'Recall': recall_score(self.y_test, y_pred, pos_label='Good'),        # Specify pos_label
                'F1 Score': f1_score(self.y_test, y_pred, pos_label='Good'),          # Specify pos_label
                'ROC-AUC': roc_auc_score(self.y_test, y_prob),
            }

    
    def interpret_with_shap(self):
        """ Interpret the trained model using SHAP values. """
        # Train a SHAP explainer for the Random Forest model
        explainer = shap.TreeExplainer(self.models['Random Forest'])
        shap_values = explainer.shap_values(self.X_test)

        # Ensure the SHAP values match the feature set shape
        # Slice the SHAP values to exclude any extra column if necessary
        if len(shap_values) == 2:  # Binary classification case
            shap_values_class_1 = shap_values[1]
        else:
            shap_values_class_1 = shap_values  # Handle case with one output

        # Check if the shapes match
        if shap_values_class_1.shape[1] != self.X_test.shape[1]:
            raise ValueError("SHAP values shape does not match X_test shape.")

        # Global summary plot (Feature importance)
        plt.figure()
        shap.summary_plot(shap_values_class_1, self.X_test, plot_type="bar")
        plt.savefig('../Images/shap_global_summary_rf.png')
        plt.show()

        # Local explanation for one instance (Example: first instance in X_test)
        shap.force_plot(explainer.expected_value[1], shap_values_class_1[0], self.X_test.iloc[0], matplotlib=True)
        plt.savefig('../Images/shap_local_explanation_rf.png')
        plt.show()

    def display_results(self):
        """ Display the results of the model evaluations """
        for model_name, metrics in self.results.items():
            print(f"Results for {model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            print("\n" + "-" * 30 + "\n")