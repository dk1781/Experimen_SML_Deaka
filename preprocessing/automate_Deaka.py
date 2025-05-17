import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocessing_data(file_path, output_path):
    try:
        # Baca data
        df = pd.read_csv(file_path)
        
        # 1. Penanganan Missing Values untuk semua kolom numerik
        df_clean = df.copy()
        numerical_features = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_features = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        # Handle nilai 0 di 'Cholesterol' dan RestingBP
        df_clean['Cholesterol'] = df_clean['Cholesterol'].replace(0, np.nan)
        df_clean['RestingBP'] = df_clean['RestingBP'].replace(0, np.nan)
        # Isi missing values untuk semua kolom numerik
        for col in numerical_features:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # 2. Standarisasi fitur numerik (kecuali target)
        if 'HeartDisease' in numerical_features:
            numerical_features.remove('HeartDisease')  # Asumsi HeartDisease adalah target
        scaler = StandardScaler()
        df_clean[numerical_features] = scaler.fit_transform(df_clean[numerical_features])
        
        # 3. Label Encoding untuk fitur kategorikal
        label_encoders = {}  # Untuk menyimpan encoder (opsional)
        for col in categorical_features:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
            label_encoders[col] = le  # Simpan encoder jika diperlukan di produksi
        
        # 4. Simpan hasil ke CSV
        df_clean.to_csv(output_path, index=False)
        print(f"Data berhasil disimpan di: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    
    preprocessing_data(args.input_path, args.output_path)
