import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocessing_data(file_path):
    # Baca data
    df = pd.read_csv(file_path)
    
    # 1. Handle missing values
    df_clean = df.copy()
    numerical_features = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    df_clean['Cholesterol'] = df_clean['Cholesterol'].replace(0, np.nan)
    df_clean['RestingBP'] = df_clean['RestingBP'].replace(0, np.nan)
    
    for col in numerical_features:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # 2. Standardize numerical features
    if 'HeartDisease' in numerical_features:
        numerical_features.remove('HeartDisease')
    scaler = StandardScaler()
    df_clean[numerical_features] = scaler.fit_transform(df_clean[numerical_features])
    
    # 3. Encode categorical features
    for col in categorical_features:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
    
    return df_clean
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True) 
    args = parser.parse_args()
    
    df = preprocessing_data(args.input_path)
    df.to_csv(args.output_path, index=False) 
