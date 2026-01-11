import pandas as pd

TARGET_COL = "Class"

def load_data(path):
    return pd.read_csv(path)

def validate_schema(df):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column `{TARGET_COL}` not found")

def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    return df

if __name__ == "__main__":
    df = load_data("data/raw/creditcard_2023.csv")
    validate_schema(df)
    df = clean_data(df)
    df.to_csv("data/processed/validated_data.csv", index=False)

