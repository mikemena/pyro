import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

def preprocess_dates(df, date_column):
    # extract relevant features
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['is_weekend'] = (df[date_column].dt.dayofweek >= 5).astype(int)

    # add cyclical encoding for months
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # drop original date column
    df = df.drop(date_column, axis=1)
    return df

def split_data(X_tensor, test_size=0.2, val_size=0.2):
    # First split: train + val vs test
    X_temp, X_test = train_test_split(X_tensor, test_size=test_size, random_state=42)

    # Second split: train vs val
    X_train, X_val = train_test_split(X_temp, test_size=val_size/(1-test_size), random_state=42)

    return X_train, X_val, X_test

def full_preprocessing(df, is_training=True, scaler=None):
    """Complete preprocessing pipeline - used by both train and predict"""
    # process date BEFORE one-hot encoding
    df = preprocess_dates(df, 'date_sent')

    # now do one-hot encoding on categorical columns
    df_encoded = pd.get_dummies(df, columns=['region','specialty'], dtype=int)
    print(df_encoded)

    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_encoded.values)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit_scaler=False")
        X_scaled = scaler.transform(df_encoded.values)

    return torch.FloatTensor(X_scaled), df_encoded.columns.tolist(), scaler

def load_and_preprocess_training_data(file_path):
    """For training"""
    # load data
    df = pd.read_excel(file_path)

    X_tensor, feature_names, scaler = full_preprocessing(df, is_training=True)

    # Split data
    X_train, X_val, X_test = split_data(X_tensor)

    return X_train, X_val, X_test, feature_names

def load_and_preprocess_prediction_data(file_path):
    """For prediction - uses scaler"""
    # load data
    df = pd.read_excel(file_path)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return full_preprocessing(df, is_training=False, scaler=scaler)
