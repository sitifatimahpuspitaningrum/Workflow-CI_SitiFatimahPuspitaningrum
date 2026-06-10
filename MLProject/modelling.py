import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, 
                             confusion_matrix, classification_report)
import mlflow
import mlflow.sklearn
import os

# Setup MLflow
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Stroke_Prediction")

# Load data
df = pd.read_csv("stroke_preprocessing.csv")
X = df.drop(columns=['stroke'])
y = df['stroke']

# Split sesuai urutan (80% train, 20% test)
split_idx = int(len(df) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# Training dengan MLflow autolog
with mlflow.start_run(run_name="RandomForest_baseline"):
    mlflow.autolog()
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("=== Hasil Training ===")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))