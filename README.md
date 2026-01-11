# Fraud Detection: From Data Analytics to End-to-End ML Pipeline

## 📌 Project Overview

This project demonstrates the **evolution of a fraud detection solution** from a traditional **data analytics and modeling approach** into a **fully automated, production-style machine learning pipeline with real-time inference**.

I initially approached fraud detection as a **data analytics problem**, focusing on understanding transaction behavior, performing feature engineering, and building a supervised classification model to identify fraudulent transactions.

As I progressed, I realized that **real-world machine learning systems do not stop at model training**. To reflect how ML is used in production, I extended the same project into an **end-to-end automated ML pipeline**, including model training, evaluation, and deployment via a REST API.

---

## 💡 Idea Behind the Project

### Initial Phase – Data Analytics Focus
- Explored historical credit card transaction data
- Performed exploratory data analysis (EDA) to understand fraud patterns
- Applied feature engineering and preprocessing
- Trained and evaluated a fraud classification model
- Analyzed class imbalance and rare-event prediction challenges

### Evolution into an ML Pipeline
While working on the analytics phase, I identified a gap between **model experimentation** and **real-world deployment**.  
This led me to redesign the project with the goal of:

- Automating the entire ML workflow
- Making the solution reproducible and scalable
- Simulating how ML models are deployed and consumed in real systems

---

## 🏗️ System Architecture

Data Ingestion
↓
Data Preprocessing
↓
Exploratory Data Analysis (EDA)
↓
Model Training
↓
Model Evaluation
↓
Model Persistence
↓
FastAPI Inference Service


Each stage is modularized and can be executed independently or as part of a single automated pipeline.

Data Source: The dataset was collected from credit card transactions made by European cardholders in 2023, with sensitive information removed to ensure privacy and compliance with ethical guidelines. https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
---

## ⚙️ Tech Stack

- **Programming Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Model Persistence:** Joblib  
- **API Framework:** FastAPI  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment Ready:** Docker (optional)  

---

## 🚀 Key Features

- Modular and automated ML pipeline
- End-to-end execution using a single command
- Automated preprocessing and scaling
- Fraud probability prediction instead of hard labels
- Threshold-based decisioning for business flexibility
- REST API for real-time inference
- Swagger UI for easy testing and validation

---

## 🔍 Fraud Detection Logic

- The model performs **binary classification**:
  - `0` → Not Fraud
  - `1` → Fraud
- Instead of returning only class labels, the API returns:
  - **Fraud probability**
  - **Decision threshold used**
  - **Final fraud prediction**

This design reflects how **real financial systems use risk scores rather than absolute decisions**.

---

## 📡 API Example Output

```json
{
  "fraud_probability": 0.105,
  "threshold_used": 0.5,
  "fraud_prediction": 0
}
```
- Fraud probability represents estimated risk

- Threshold controls sensitivity to fraud detection

- Decision changes dynamically based on business requirements

🧪 Threshold Tuning (Why It Matters)

Fraud is a rare event, so using a fixed threshold (e.g., 0.5) is often suboptimal.

Lower thresholds:

- Catch more frauds

- Increase false positives

Higher thresholds:

- Reduce false positives

- May miss some fraud cases

The API supports dynamic threshold tuning to simulate real-world decision strategies.

▶️ How to Run the Project Locally
```
git clone <repository-url>
cd fraud-detection-ml-pipeline

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python run_pipeline.py
uvicorn app:app --reload
```
Open in browser: http://127.0.0.1:8000/docs

📂 Project Structure

fraud-detection-ml-pipeline/
│
├── data/
├── eda/
├── logs/
├── models/
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train.py
│   └── evaluate.py
│
├── app.py
├── run_pipeline.py
├── requirements.txt
├── Dockerfile
└── README.md

📈 What This Project Demonstrates

- Transition from data analytics to ML engineering

- Understanding of the full ML lifecycle

- Ability to operationalize ML models

- Practical handling of rare-event classification

- API-based ML inference design

- Production-oriented thinking

🔮 Future Improvements

- Model explainability (SHAP / feature importance)

- Monitoring & logging of predictions

- Cloud deployment (AWS / Render / Railway)

- Streaming or batch inference

- Feedback loop for model retraining

🧠 Key Learning Outcome

This project helped me bridge the gap between exploratory data analysis and production-grade machine learning systems, highlighting how analytical models are transformed into deployable and scalable ML solutions.









