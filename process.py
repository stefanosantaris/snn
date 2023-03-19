import pandas as pd
import os 
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.model_selection import train_test_split


def df_format(df, train_path, test_path):

    # Apply label encoder
    df = df.apply(LabelEncoder().fit_transform)

    # Split dataset to test/train
    train_df, test_df = train_test_split(df, test_size=0.2)

    # Scale train/test data
    scaler = MaxAbsScaler()
    scaler.fit(train_df)
    scaled = scaler.transform(train_df)
    scaled_train_df = pd.DataFrame(scaled, columns=train_df.columns)

    scaler.fit(test_df)
    scaled = scaler.transform(test_df)
    scaled_test_df = pd.DataFrame(scaled, columns=test_df.columns)

    scaled_train_df.to_csv(train_path, index=False)
    scaled_test_df.to_csv(test_path, index=False)

print("1. [Criteo] Generating a sample")
os.system("head -n 500000 .tmp/criteo_dataset.csv > .tmp/sample_criteo_dataset.csv")

print("2. [Criteo] Reshaping dataframe")
df = pd.read_csv(".tmp/sample_criteo_dataset.csv")
cols = ['visit', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'treatment', 'conversion', 'exposure']
df = df.reindex(columns=cols)
df.to_csv(".tmp/sample_criteo_dataset.csv", index=False)

print("3. [Criteo] Preprocessing dataframe")
df_format(df, '.tmp/sample_train_criteo_dataset.csv', '.tmp/sample_test_criteo_dataset.csv')

print("4. [Avazu] Generating a sample")
os.system("head -n 500000 .tmp/avazu_dataset.csv > .tmp/sample_avazu_dataset.csv")

print("5. [Avazu] Reshaping dataframe")
df = pd.read_csv(".tmp/sample_avazu_dataset.csv")
df = df.drop(['id'], axis=1)
df.to_csv(".tmp/sample_avazu_dataset.csv", index=False)

print("6. [Avazu] Preprocessing dataframe")
df_format(df, '.tmp/sample_train_avazu_dataset.csv', '.tmp/sample_test_avazu_dataset.csv')


