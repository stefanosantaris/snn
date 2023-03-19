import pandas as pd
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
from autofeat import AutoFeatRegressor

def process():
    
    # Read csv
    print("Reading csv...")
    df = pd.read_csv(".tmp/dataset_all_samples.csv", sep=";")

    # Fill null values
    df = df.fillna(method="ffill")
  
    # Transform every value to numeric
    print("Transforming to numerical...")
    df = df.apply(LabelEncoder().fit_transform)

    # Remove outliers 
    # z_scores = stats.zscore(df)
    # abs_z_scores = np.abs(z_scores)
    # filtered_entries = (abs_z_scores < 3).all(axis=1)
    # df = df[filtered_entries]
        

    # Scale data
    print("Scaling the data...")
    scaler = MaxAbsScaler()
    scaler.fit(df)
    scaled_df = scaler.transform(df)
    df = pd.DataFrame(scaled_df, columns=df.columns)
    
    y = df[['click']]
    X = df[['userID','itemID','category','len_user_items','len_user_categories','len_item_users']]

    model = AutoFeatRegressor()
    
    X_tran = model.fit_transform(X.to_numpy(), y.to_numpy().flatten())
    df = pd.DataFrame(X_tran)
    df.insert(loc=0, column="click", value=y)

    # Split dataset to train/test
    print("Spliting to train/test....")
    train_df, test_df = train_test_split(df, test_size=0.2)

    print("Storing to csv...")
    train_df.to_csv(".tmp/dataset_train.csv", index=False)
    test_df.to_csv(".tmp/dataset_test.csv", index=False)

if __name__ == "__main__":
    process()
