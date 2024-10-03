import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

class TransactionProcessor:
    def __init__(self, df):
        """
        Initialize the TransactionProcessor class with a DataFrame.
        """
        self.df = df

    def convert_data_types(self):
        """
        Ensure the necessary columns are converted to numeric data types.
        This will help avoid errors during aggregate functions or scaling.
        """
        # Convert 'Amount' and 'Value' to numeric, coerce errors to NaN
        self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce')
        self.df['Value'] = pd.to_numeric(self.df['Value'], errors='coerce')

    def aggregate_features(self):
        """
        Create aggregate features for each customer:
        - Total Transaction Amount
        - Average Transaction Amount
        - Transaction Count
        - Standard Deviation of Transaction Amounts
        """
        agg_features = self.df.groupby('CustomerId').agg(
            total_transaction_amount=('Amount', 'sum'),
            average_transaction_amount=('Amount', 'mean'),
            transaction_count=('TransactionId', 'count'),
            std_transaction_amount=('Amount', 'std')
        ).reset_index()

        # Merge aggregate features back into the original dataframe
        self.df = pd.merge(self.df, agg_features, on='CustomerId', how='left')

        return self.df

    def extract_time_features(self):
        """
        Extract features such as:
        - Transaction Hour
        - Transaction Day
        - Transaction Month
        - Transaction Year
        """
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        self.df['transaction_hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['transaction_day'] = self.df['TransactionStartTime'].dt.day
        self.df['transaction_month'] = self.df['TransactionStartTime'].dt.month
        self.df['transaction_year'] = self.df['TransactionStartTime'].dt.year

        return self.df

    def encode_categorical_variables(self, label_encode_cols=None, one_hot_encode_cols=None):
        """
        Encode categorical variables:
        - Label Encoding: Assign unique integer values to categories.
        - One-Hot Encoding: Convert categorical values into binary vectors.
        """
        # Label encoding for specified columns
        if label_encode_cols:
            label_encoder = LabelEncoder()
            for col in label_encode_cols:
                self.df[col] = label_encoder.fit_transform(self.df[col])

        # One-Hot encoding for specified columns
        if one_hot_encode_cols:
            self.df = pd.get_dummies(self.df, columns=one_hot_encode_cols)

        return self.df

  

    def scale_numerical_features(self, columns, method='normalize'):
        """
        Scale numerical features using:
        - Normalization: Scale to [0, 1].
        - Standardization: Scale to mean 0 and standard deviation 1.
        """
        if method == 'normalize':
            scaler = MinMaxScaler()
        elif method == 'standardize':
            scaler = StandardScaler()

        self.df[columns] = scaler.fit_transform(self.df[columns])

        return self.df

    def process_data(self):
        """
        Main function to process the data in sequence:
        1. Convert data types
        2. Aggregate features
        3. Extract time features
        4. Encode categorical variables
        5. Handle missing values
        6. Normalize/Standardize numerical features
        7. Add all new features to the final DataFrame
        """
        # Step 1: Convert Data Types
        self.convert_data_types()

        # Step 2: Aggregate Features
        self.df = self.aggregate_features()

        # Step 3: Extract Time Features
        self.df = self.extract_time_features()

        # Step 4: Encode Categorical Variables
        self.df = self.encode_categorical_variables(
            label_encode_cols=['ProductCategory'],
            one_hot_encode_cols=['CurrencyCode']
        )

        # Step 5: Handle Missing Values
        #self.df = self.handle_missing_values(method='impute', strategy='mean')

        # Step 6: Scale Numerical Features
        self.df = self.scale_numerical_features(columns=['Amount', 'Value'], method='standardize')

        return self.df

    def save_cleaned_data(self, file_path):
        """
        Save the processed data back to a CSV file.
        """
        self.df.to_csv(file_path, index=False)
        print(f"Cleaned data saved to {file_path}")