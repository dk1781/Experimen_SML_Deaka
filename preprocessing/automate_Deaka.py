# File: preprocessing/automate_Deaka.py
import pandas as pd
import numpy as np
import argparse
import os
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocessing_data(file_path):
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
    
    # Impute median
    for col in numerical_features:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # 2. Standardize numerical features (kecuali target)
    if 'HeartDisease' in numerical_features:
        numerical_features.remove('HeartDisease')
    scaler = StandardScaler()
    df_clean[numerical_features] = scaler.fit_transform(df_clean[numerical_features])
    
    # 3. Encode categorical features
    for col in categorical_features:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    
    return df_clean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate preprocessing heart disease dataset"
    )
    parser.add_argument(
        '--input_path', '-i', type=str, required=True,
        help='Path ke file CSV mentah (raw)'
    )
    parser.add_argument(
        '--output_path', '-o', type=str, required=True,
        help='Path untuk menyimpan file CSV hasil preprocessing'
    )
    args = parser.parse_args()
    
    # Validasi input
    if not os.path.isfile(args.input_path):
        print(f"Error: input file '{args.input_path}' tidak ditemukan.", file=sys.stderr)
        sys.exit(1)
    
    # Buat direktori output jika belum ada
    out_dir = os.path.dirname(args.output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Jalankan preprocessing
    df = preprocessing_data(args.input_path)
    df.to_csv(args.output_path, index=False)
