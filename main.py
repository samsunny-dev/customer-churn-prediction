import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

customer_df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
customer_df.head()

customer_df['TotalCharges'] = pd.to.numeric(customer_df['TotalCharges'], errors='coerce')
customer_df.dropna(inplace=True)
customer_df(['customerID'], axis=1, inplace=True)
customer_df['Churn'] = LabelEncoder().fit_transform(customer_df['Churn'])
customer_df = pd.get_dummies(customer_df, drop_first=True)

X = customer_df.drop('Churn', axis=1)
y = customer_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)

lr.fit(X_train_res, y_train_res)
rf.fit(X_train_res, y_train_res)

for model in [lr, rf]:
    print(f"\n--- {model.__class__.__name__} ---")
    y_preds = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_preds))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba)}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {roc_auc_score(y_test, y_proba):.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()