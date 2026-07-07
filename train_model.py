import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import os

print("Loading data...")
train_transaction = pd.read_csv("data/train_transaction.csv")
train_identity = pd.read_csv("data/train_identity.csv")

print("Merging datasets...")
df = train_transaction.merge(train_identity, on="TransactionID", how="left")

print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")

# Drop high-cardinality / leaky columns
drop_cols = ["TransactionID", "TransactionDT"]
df = df.drop(columns=drop_cols)

# Encode categoricals
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes

# Fill missing values
df = df.fillna(-999)

# Split features and target
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

print("\nEvaluating...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Save model and feature names
os.makedirs("model", exist_ok=True)
with open("model/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

feature_names = X.columns.tolist()
with open("model/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\nModel saved to model/xgb_model.pkl")
print("Training complete!")