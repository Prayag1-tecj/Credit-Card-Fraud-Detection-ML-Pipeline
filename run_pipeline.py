import os

os.system("python SRC/data_ingestion.py")
os.system("python SRC/preprocessing.py")
os.system("python SRC/eda.py")
os.system("python SRC/train.py")
os.system("python SRC/evaluate.py")

print("✅ End-to-End Fraud Detection Pipeline Executed Successfully")
