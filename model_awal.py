# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv(
    '/content/drive/MyDrive/machine learning/healthcare-dataset-stroke-data.csv',
    sep=';'
)
data

target = "stroke"
X = data.drop(columns=[target])
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

baseline_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', LogisticRegression(
        max_iter=300,
        class_weight='balanced',
        random_state=42
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, pos_label=1),
    'recall': make_scorer(recall_score, pos_label=1),
    'f1': make_scorer(f1_score, pos_label=1),
    'roc_auc': 'roc_auc'
}

cv_result = cross_validate(
    baseline_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=False
)

print("\nBaseline Model\n")

print("Per-Fold Metrics:")
for i in range(cv.get_n_splits()):
    print(
        f"Fold {i+1}: "
        f"Acc={cv_result['test_accuracy'][i]:.4f}, "
        f"Prec={cv_result['test_precision'][i]:.4f}, "
        f"Recall={cv_result['test_recall'][i]:.4f}, "
        f"F1={cv_result['test_f1'][i]:.4f}, "
        f"AUC={cv_result['test_roc_auc'][i]:.4f}"
    )

print("\nMean ± Std:")
for metric in scoring.keys():
    mean = cv_result[f'test_{metric}'].mean()
    std = cv_result[f'test_{metric}'].std()
    print(f"{metric:10s}: {mean:.4f} ± {std:.4f}")
