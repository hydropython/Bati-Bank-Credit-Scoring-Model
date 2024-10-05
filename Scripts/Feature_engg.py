import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

class TransactionProcessor:
    def __init__(self, df):
        self.df = df
    
    def aggregate_features(self):
        """
        Create aggregate features for each customer:
        - Total Transaction Amount
        - Average Transaction Amount
        - Transaction Count
        - Standard Deviation of Transaction Amounts
        """
        print("\nAggregating features per customer...")
        agg_features = self.df.groupby('CustomerId').agg(
            total_transaction_amount=('Amount', 'sum'),
            average_transaction_amount=('Amount', 'mean'),
            transaction_count=('TransactionId', 'count'),
            std_transaction_amount=('Amount', 'std')
        ).reset_index()

        # Merge aggregate features back into the original dataframe
        self.df = pd.merge(self.df, agg_features, on='CustomerId', how='left')
        print("Data after aggregation:\n", self.df.head())
        return self.df

    def extract_time_features(self):
        """
        Extract features such as:
        - Transaction Hour
        - Transaction Day
        - Transaction Month
        - Transaction Year
        """
        print("\nExtracting time features...")
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        self.df['transaction_hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['transaction_day'] = self.df['TransactionStartTime'].dt.day
        self.df['transaction_month'] = self.df['TransactionStartTime'].dt.month
        self.df['transaction_year'] = self.df['TransactionStartTime'].dt.year
        
        print("Data after extracting time features:\n", self.df.head())
        return self.df

    def encode_categorical_columns(self):
        """
        Encode categorical variables using Label Encoding.
        """
        print("\nEncoding categorical columns...")
        # Define categorical columns to encode
        categorical_columns = ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 
                            'ChannelId', 'PricingStrategy']  # Included PricingStrategy

        # Apply label encoding to each categorical column
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            if self.df[col].dtype == 'object' or self.df[col].dtype == 'category':  # Check if it's categorical
                self.df[col] = label_encoder.fit_transform(self.df[col])

        print("Data after encoding categorical columns:\n", self.df[categorical_columns].head())
        return self.df

    def handle_missing_values(self):
        """
        Handle missing values using imputation or removal.
        """
        print("\nHandling missing values...")
        # Imputation: Filling missing values with mean for numerical features
        for col in self.df.select_dtypes(include=['float64', 'int64']).columns:
            self.df[col].fillna(self.df[col].mean(), inplace=True)
        
        print("Data after handling missing values:\n", self.df.head())
        return self.df

    def normalize_or_standardize(self, method='normalize'):
        """
        Normalize or standardize numerical features:
        - Normalization: Scales the data to a range of [0, 1].
        - Standardization: Scales the data to have a mean of 0 and a standard deviation of 1.
        """
        print(f"\nApplying {method} to numerical columns...")
        
        # Define numerical columns to scale
        numerical_columns = ['Amount', 'Value', 'total_transaction_amount', 
                             'average_transaction_amount', 'transaction_count', 
                             'std_transaction_amount']

        if method == 'normalize':
            scaler = MinMaxScaler()
        elif method == 'standardize':
            scaler = StandardScaler()
        else:
            raise ValueError("Method must be 'normalize' or 'standardize'")
        
        # Apply the scaler
        self.df[numerical_columns] = scaler.fit_transform(self.df[numerical_columns])

        print(f"Data after {method}:\n", self.df[numerical_columns].head())
        return self.df
    
    def save_final_dataframe(self, filename='final.csv'):
        """
        Save the final DataFrame to a CSV file.
        """
        self.df.to_csv(filename, index=False)
        print(f"Final DataFrame saved to {filename}.")