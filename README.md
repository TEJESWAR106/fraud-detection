# Credit Card Fraud Detection

An end-to-end machine learning project that detects fraudulent credit card transactions using XGBoost and anomaly detection, with an interactive Streamlit dashboard.

---

## Live Demo

[![Open in Streamlit](https://fraud-detection-xzrevbrzwmoup7kzrfrktv.streamlit.app/)


---

## Overview

Credit card fraud is a severe problem in the fintech industry. This project builds a complete ML pipeline — from raw imbalanced data to a deployed web app — that flags fraudulent transactions in real time.

- Dataset: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, only 0.17% fraudulent)
- Handles extreme class imbalance using **SMOTE**
- Achieves **ROC-AUC > 0.98** with XGBoost

---

## Project Structure

```
fraud_detection/
├── train.py              # Model training, evaluation, and saving
├── app.py                # Streamlit dashboard
├── fraud_model.pkl       # Saved XGBoost model
├── features.pkl          # Saved feature column names
├── confusion_matrix.png  # Evaluation output
├── roc_curve.png         # Evaluation output
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Model | XGBoost, Isolation Forest |
| Preprocessing | Scikit-learn, imbalanced-learn (SMOTE) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## How It Works

### 1. Data Preprocessing
- Scales `Amount` and `Time` columns using `StandardScaler`
- Removes duplicate rows
- Splits into 80% train / 20% test with stratification

### 2. Handling Class Imbalance
- Applies **SMOTE** (Synthetic Minority Oversampling Technique) on training data only
- Balances fraud vs legitimate transaction ratio before model fitting

### 3. Model Training
- **XGBoost Classifier** — primary supervised model (200 estimators, max depth 4)
- **Isolation Forest** — unsupervised anomaly detection baseline for comparison

### 4. Evaluation
- ROC-AUC score (target > 0.98)
- Precision, Recall, F1-score (focus on recall for fraud class)
- Confusion matrix
- ROC curve

### 5. Streamlit Dashboard
- Upload any CSV of transactions
- Fraud rows highlighted in red
- Displays fraud count, rate metrics
- Shows ROC curve and confusion matrix charts (if labels available)

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | > 0.98 |
| Fraud Recall | > 0.90 |
| Overall Accuracy | > 99% |

---

## Getting Started Locally

### 1. Clone the repo
```bash
git clone https://github.com/TEJESWAR106/fraud-detection.git
cd fraud-detection
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project folder.

### 5. Train the model
```bash
python train.py
```

### 6. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.


---

## Internship

This project was completed as part of the **Elevate Labs AI & ML Internship** (2-week project phase).

---

## License

This project is for educational purposes only.
