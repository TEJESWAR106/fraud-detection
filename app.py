import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)

model    = joblib.load('fraud_model.pkl')
features = joblib.load('features.pkl')

st.set_page_config(page_title='Fraud Detector', layout='wide')
st.title('Credit Card Fraud Detector')
st.write('Upload a CSV of transactions to detect fraud instantly.')

uploaded = st.file_uploader('Upload transactions CSV', type='csv')

if uploaded:
    df = pd.read_csv(uploaded)

    # Scale Amount & Time (same as training)
    df['scaled_amount'] = StandardScaler().fit_transform(df[['Amount']])
    df['scaled_time']   = StandardScaler().fit_transform(df[['Time']])
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    X = df[features]
    df['Prediction'] = model.predict(X)
    df['Fraud_Prob']  = model.predict_proba(X)[:,1].round(3)

    fraud_count = int(df['Prediction'].sum())
    total       = len(df)

    col1, col2, col3 = st.columns(3)
    col1.metric('Total Transactions', total)
    col2.metric('Fraudulent', fraud_count)
    col3.metric('Fraud Rate', f"{round(fraud_count/total*100, 2)}%")

    st.subheader('Transactions (fraud rows in red)')

    def color_row(row):
        bg = 'background-color:#fee2e2' if row['Prediction']==1 else ''
        return [bg]*len(row)

    st.dataframe(df.style.apply(color_row, axis=1), use_container_width=True)

    # If original labels exist, show charts
    if 'Class' in df.columns:
        st.subheader('Model Evaluation')
        auc = roc_auc_score(df['Class'], df['Fraud_Prob'])
        st.write(f"ROC-AUC: **{round(auc, 4)}**")

        fig1, ax1 = plt.subplots()
        fpr, tpr, _ = roc_curve(df['Class'], df['Fraud_Prob'])
        ax1.plot(fpr, tpr); ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
        ax1.set_title('ROC Curve')

        fig2, ax2 = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            df['Class'], df['Prediction'], ax=ax2)

        col_a, col_b = st.columns(2)
        col_a.pyplot(fig1)
        col_b.pyplot(fig2)