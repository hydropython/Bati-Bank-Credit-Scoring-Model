import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# FastAPI request model for hyperparameter tuning request
class HyperparameterRequest(BaseModel):
    test_size: float = 0.2
    random_state: int = 42

class ModelSelector:
    def __init__(self, data):
        self.data = data
        self.target_column = 'UserLabel'  # Set target variable
        # Updated feature columns based on the provided columns
        self.feature_columns = [
            'PricingStrategy', 'total_transaction_amount', 'average_transaction_amount',
            'transaction_count', 'transaction_hour', 'transaction_day',
            'transaction_month', 'transaction_year'
        ]
        self.models = {
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier()
        }
        self.results = {}

    def preprocess_data(self):
        """ Preprocess the data by handling non-numeric columns and missing values """
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns

        self.data.dropna(axis=1, how='all', inplace=True)

        numeric_cols = self.data.select_dtypes(include=['number']).columns
        imputer_numeric = SimpleImputer(strategy='mean')
        imputed_numeric_data = imputer_numeric.fit_transform(self.data[numeric_cols])
        self.data[numeric_cols] = pd.DataFrame(imputed_numeric_data, columns=numeric_cols)

        for col in categorical_cols:
            encoder = LabelEncoder()
            self.data[col] = encoder.fit_transform(self.data[col].astype(str))

        imputer_categorical = SimpleImputer(strategy='most_frequent')
        imputed_categorical_data = imputer_categorical.fit_transform(self.data[categorical_cols])
        self.data[categorical_cols] = pd.DataFrame(imputed_categorical_data, columns=categorical_cols)

    def split_data(self, test_size=0.2, random_state=42):
        """ Split the data into training and testing sets """
        self.preprocess_data()
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
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search_rf = GridSearchCV(estimator=self.models['Random Forest'], param_grid=rf_param_grid, scoring='accuracy', cv=5)
        grid_search_rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = grid_search_rf.best_estimator_

        for params, score in zip(grid_search_rf.cv_results_['params'], grid_search_rf.cv_results_['mean_test_score']):
            hyperparameter_results.append({**params, 'Model': 'Random Forest', 'Score': score})

        gb_param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        grid_search_gb = GridSearchCV(estimator=self.models['Gradient Boosting'], param_grid=gb_param_grid, scoring='accuracy', cv=5)
        grid_search_gb.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = grid_search_gb.best_estimator_

        for params, score in zip(grid_search_gb.cv_results_['params'], grid_search_gb.cv_results_['mean_test_score']):
            hyperparameter_results.append({**params, 'Model': 'Gradient Boosting', 'Score': score})

        hyperparameter_df = pd.DataFrame(hyperparameter_results)
        score_column = hyperparameter_df.pop('Score')
        hyperparameter_df['Score'] = score_column

        return hyperparameter_df

    def plot_hyperparameter_results(self, hyperparameter_df):
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
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            self.results[model_name] = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred, pos_label='Good'),
                'Recall': recall_score(self.y_test, y_pred, pos_label='Good'),
                'F1 Score': f1_score(self.y_test, y_pred, pos_label='Good'),
                'ROC-AUC': roc_auc_score(self.y_test, y_prob),
            }
        return self.results

@app.post("/hyperparameter-tuning/")
def hyperparameter_tuning(request: HyperparameterRequest):
    # Example: Assuming `data` is already loaded as a pandas DataFrame
    data = pd.read_csv("../Data/sample_data.csv")
    model_selector = ModelSelector(data)
    model_selector.split_data(test_size=request.test_size, random_state=request.random_state)
    hyperparameter_df = model_selector.hyperparameter_tuning()
    
    return hyperparameter_df.to_dict()

@app.get("/evaluate-models/")
def evaluate_models():
    data = pd.read_csv("../Data/sample_data.csv")
    model_selector = ModelSelector(data)
    model_selector.split_data()
    model_selector.train_models()
    results = model_selector.evaluate_models()

    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)