import pandas as pd
import numpy as np
import argparse
import os
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocessing_data(file_path, output_path):
    # Baca data
    df = pd.read_csv(file_path)
    
    # 1. Handle missing values
    df_clean = df.copy()
    numerical_features = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Ganti 0 dengan NaN untuk kolom tertentu
    for col_zero in ['Cholesterol', 'RestingBP']:
        if col_zero in df_clean.columns:
            df_clean[col_zero] = df_clean[col_zero].replace(0, np.nan)
    
    # Impute mean
    for col in numerical_features:
        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    # 2. Standardize numerical features (kecuali target)
    if 'HeartDisease' in numerical_features:
        numerical_features.remove('HeartDisease')
    scaler = StandardScaler()
    df_clean[numerical_features] = scaler.fit_transform(df_clean[numerical_features])
    
    # 3. Encode categorical features
    for col in categorical_features:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    #4 Split Dataset
    X=df_clean.drop(columns=['HeartDisease'])
    y=df_clean['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    output_path = "dataset_preprocessing"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    X_train.to_csv(os.path.join(output_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_path, "y_test.csv"), index=False)
    
    return df_clean,X_train, X_test, y_train, y_test

if __name__ == "__main__":
    input_file = "rawdata/heart.csv"
    output_path = "preprocessing/outputs"

    try:
        mlflow.set_tracking_uri("file:./mlruns")

        with mlflow.start_run(run_name="Preprocessing_Run"):
            df_clean, X_train, X_test, y_train, y_test = preprocess_dataset(input_file, output_path)

            mlflow.log_param("input_file", input_file)
            mlflow.log_param("output_path", output_path)
            mlflow.log_metric("rows_after_cleaning", df_clean.shape[0])

            mlflow.log_artifact(os.path.join(output_path, 'X_train.csv'))
            mlflow.log_artifact(os.path.join(output_path, 'X_test.csv'))
            mlflow.log_artifact(os.path.join(output_path, 'y_train.csv'))
            mlflow.log_artifact(os.path.join(output_path, 'y_test.csv'))
   

    except Exception as err:
        print(f"[ERROR] Tahap Preprocessing failed: {err}")
