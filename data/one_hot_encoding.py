import pandas as pd
import numpy as np
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

# load data
df = pd.read_excel('cred_data.xlsx')
# print(df)

# process date BEFORE one-hot encoding
df = preprocess_dates(df, 'date_sent')

# now do one-hot encoding on categorical columns
df_encoded = pd.get_dummies(df, columns=['region','specialty'], dtype=int)
print(df_encoded)

# feature_columns = [col for col in df_encoded.columns]
# X = df_encoded[feature_columns].values

# Convert to tensor
X_tensor = torch.FloatTensor(df_encoded.values)

input_size = X_tensor.shape[1]
print(f"Input size: {input_size}")
