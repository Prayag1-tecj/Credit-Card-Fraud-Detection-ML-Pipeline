import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

TARGET_COL = "Class"

def preprocess(df):
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "models/scaler.pkl")

    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df[TARGET_COL] = y.values

    return processed_df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/validated_data.csv")
    processed_df = preprocess(df)
    processed_df.to_csv("data/processed/processed_data.csv", index=False)
