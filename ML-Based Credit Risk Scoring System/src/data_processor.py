import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        return pd.read_csv(file_path)
    
    def preprocess_data(self, df):
        """Preprocess the data including handling missing values and encoding"""
        # Separate numerical and categorical columns
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Handle missing values for numerical columns
        if len(numerical_columns) > 0:
            df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
        
        # Handle missing values for categorical columns
        if len(categorical_columns) > 0:
            df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        
        # Convert categorical variables
        df = pd.get_dummies(df, columns=categorical_columns)
        
        return df
    
    def prepare_features(self, df, target_column):
        """Prepare features and target for model training"""
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)