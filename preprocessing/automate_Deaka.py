import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocessing_data (file_path):

    df= pd.read_csv(file_path)
    #Menentukan Feature numerik dan kategorik
    
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_features.remove('HeartDisease')
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    #Penanganan inaccurate value
    df_clean = df.copy()
    df_clean['Cholesterol'] = df_clean['Cholesterol'].replace(0, np.nan)
    df_clean['Cholesterol'] = df_clean['Cholesterol'].fillna(df_clean['Cholesterol'].mean())
    
    
    #Standarisasi
    scaler = StandardScaler()
    df_clean[numerical_features] = scaler.fit_transform(df_clean[numerical_features])

    #Encode fitur kategorikal
    for col in categorical_features:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        
    return df_clean

