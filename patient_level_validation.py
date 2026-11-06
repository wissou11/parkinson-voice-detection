"""Compare record-level and patient-level validation on the Parkinson's dataset.

Each patient has 6 or 7 recordings. A random split puts recordings of the same
patient in both training and test sets, while a patient-level split keeps each
patient on one side only. This script uses the models and features of the
notebook, repeats 5-fold cross-validation 20 times for both splits, and writes
the averages to results/patient_level_validation.json.

Usage:
    python patient_level_validation.py
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

N_REPEATS = 20
JITTER_SHIMMER = [
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
]
MODELS = {
    "Logistic Regression": lambda: LogisticRegression(C=1, penalty="l1", solver="liblinear", max_iter=1000),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=1, metric="manhattan"),
    "Naive Bayes": lambda: GaussianNB(),
    "SVM": lambda: SVC(C=10, kernel="rbf", gamma="auto", probability=True, random_state=42),
}

df = pd.read_csv("parkinsons.data")
df["Freq_Range"] = df["MDVP:Fhi(Hz)"] - df["MDVP:Flo(Hz)"]
df["Freq_Ratio"] = df["MDVP:Fhi(Hz)"] / df["MDVP:Flo(Hz)"]
patients = np.array([name.rsplit("_", 1)[0] for name in df["name"]], dtype=object)
X = df.drop(columns=["name", "status"])
y = df["status"].to_numpy()


def build(model_factory):
    """PCA is fitted inside each fold, so test rows never influence it."""
    other_columns = [c for c in X.columns if c not in JITTER_SHIMMER]
    features = ColumnTransformer([
        ("jitter_shimmer", make_pipeline(StandardScaler(), PCA(n_components=2)), JITTER_SHIMMER),
        ("other", "passthrough", other_columns),
    ])
    return make_pipeline(features, StandardScaler(), model_factory())


def evaluate(model_factory, patient_level):
    scores = []
    for seed in range(N_REPEATS):
        if patient_level:
            cv, groups = StratifiedGroupKFold(5, shuffle=True, random_state=seed), patients
        else:
            cv, groups = StratifiedKFold(5, shuffle=True, random_state=seed), None
        prob = cross_val_predict(build(model_factory), X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
        pred = (prob >= 0.5).astype(int)
        scores.append({
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, prob),
            "recall": recall_score(y, pred),
            "specificity": recall_score(y, pred, pos_label=0),
        })
    return {k: round(float(np.mean([s[k] for s in scores])), 3) for k in scores[0]}


always_parkinson = np.ones_like(y)
results = {
    "n_recordings": int(len(df)),
    "n_patients": int(len(set(patients))),
    "n_patients_parkinson": int(df.groupby(patients)["status"].first().sum()),
    "baseline_always_parkinson": {
        "accuracy": round(float(accuracy_score(y, always_parkinson)), 3),
        "balanced_accuracy": 0.5,
        "f1": round(float(f1_score(y, always_parkinson)), 3),
        "roc_auc": 0.5,
        "recall": 1.0,
        "specificity": 0.0,
    },
    "models": {
        name: {
            "record_level": evaluate(factory, patient_level=False),
            "patient_level": evaluate(factory, patient_level=True),
        }
        for name, factory in MODELS.items()
    },
}

os.makedirs("results", exist_ok=True)
with open("results/patient_level_validation.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
