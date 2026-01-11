import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    ratio = df["Class"].value_counts(normalize=True)
    ratio.to_csv("eda/eda_report.txt")

    sns.countplot(x="Class", data=df)
    plt.title("Fraud vs Non-Fraud Distribution")
    plt.savefig("eda/plots/class_distribution.png")
    plt.clf()

if __name__ == "__main__":
    df = pd.read_csv("data/processed/processed_data.csv")
    run_eda(df)
