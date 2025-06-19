import pandas as pd
import numpy as np
import torch

def preprocess_dates(df, date_column):
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['is_weekend'] = (df[date_column].dt.dayofweek >= 5).astype(int)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df = df.drop(date_column, axis=1)
    return df


df = pd.read_excel('cred_data.xlsx')
# print(df)

df = preprocess_dates(df, 'date_sent')

df_encoded = pd.get_dummies(df, columns=['region','specialty'], dtype=int)
print(df_encoded)

feature_columns = [col for col in df_encoded.columns]
X = df_encoded[feature_columns].values

# Convert to tensor
X_tensor = torch.FloatTensor(X)

input_size = X_tensor.shape[1]
print(f"Input size: {input_size}")
