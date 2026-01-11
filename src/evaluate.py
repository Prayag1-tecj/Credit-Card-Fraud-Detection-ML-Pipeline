import pandas as pd
import joblib
import json
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

TARGET_COL = "Class"

def evaluate(df):
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    model = joblib.load("models/fraud_model.pkl")
    preds = model.predict(X)

    metrics = {
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1_score": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, preds)
    }

    with open("logs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    df = pd.read_csv("data/processed/processed_data.csv")
    evaluate(df)
