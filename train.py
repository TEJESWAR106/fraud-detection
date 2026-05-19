import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# 1. Load data
df = pd.read_csv('"C:\Users\TEJESWAR A\Downloads\Dataset\creditcard.csv"')
print("Shape:", df.shape)
print(df['Class'].value_counts())

# 2. Plot class imbalance
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.savefig('class_dist.png')
plt.show()

# 3. Scale Amount & Time
df['scaled_amount'] = StandardScaler().fit_transform(df[['Amount']])
df['scaled_time']   = StandardScaler().fit_transform(df[['Time']])
df.drop(['Time', 'Amount'], axis=1, inplace=True)
df.drop_duplicates(inplace=True)

# 4. Split
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print("After SMOTE:", y_res.value_counts())

# Save for app.py
joblib.dump(list(X.columns), 'features.pkl')
print("Features saved.")

from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)

# Train XGBoost
xgb = XGBClassifier(n_estimators=200, max_depth=4,
                    learning_rate=0.1, eval_metric='logloss',
                    random_state=42)
xgb.fit(X_res, y_res)

# Predict
y_pred  = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:,1]

# Metrics
print(classification_report(y_test, y_pred))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

# Confusion matrix
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
plt.savefig('confusion_matrix.png')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label='XGBoost')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curve')
plt.savefig('roc_curve.png')
plt.show()

# Save model
joblib.dump(xgb, 'fraud_model.pkl')
print("Model saved as fraud_model.pkl")