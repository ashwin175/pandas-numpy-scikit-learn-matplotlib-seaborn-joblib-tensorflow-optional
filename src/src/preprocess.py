import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    df = df.dropna()

    # Label encode categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop("label", axis=1))
    y = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    return X, y, scaler
