import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_validate, GridSearchCV
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay,
    make_scorer
)


# LOAD DATA
data = pd.read_csv(
    'data/healthcare-dataset-stroke-data.csv',
    sep=';'
)

target = 'stroke'
X = data.drop(columns=[target])
y = data[target]


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)


# PREPROCESSING PIPELINE
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

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


# CROSS VALIDATION SETUP
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, pos_label=1),
    'recall': make_scorer(recall_score, pos_label=1),
    'f1': make_scorer(f1_score, pos_label=1),
    'roc_auc': 'roc_auc'
}


# BASELINE MODEL
lr_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', LogisticRegression(
        max_iter=300,
        class_weight='balanced',
        random_state=42
    ))
])

lr_cv = cross_validate(
    lr_pipeline, X_train, y_train,
    cv=cv, scoring=scoring
)

print('\n=== Logistic Regression (Baseline) ===')
for m in scoring:
    print(f"{m}: {lr_cv['test_' + m].mean():.4f} ± {lr_cv['test_' + m].std():.4f}")


# RANDOM FOREST
rf_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    ))
])

rf_param_grid = {
    'model__n_estimators': [200, 300],
    'model__max_depth': [None, 10, 20]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)


# XGBOOST (BOOSTING WAJIB)
xgb_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        random_state=42
    ))
])

xgb_param_grid = {
    'model__n_estimators': [200, 300],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1]
}

xgb_grid = GridSearchCV(
    xgb_pipeline,
    xgb_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)

best_model = xgb_grid.best_estimator_

# FINAL EVALUATION (TEST SET)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print('\n=== FINAL TEST PERFORMANCE ===')
print('Accuracy :', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall   :', recall_score(y_test, y_pred))
print('F1-score :', f1_score(y_test, y_pred))
print('ROC-AUC  :', roc_auc_score(y_test, y_proba))

print('\nClassification Report:')
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title('Confusion Matrix - XGBoost')
plt.show()


#SHAP EXPLAINABILITY
preprocessor = best_model.named_steps['preprocess']
model = best_model.named_steps['model']

X_test_pre = preprocessor.transform(X_test)

num_cols = numeric_cols
cat_cols = preprocessor.named_transformers_['cat']['onehot'] \
    .get_feature_names_out(categorical_cols)

feature_names = np.concatenate([num_cols, cat_cols])

X_test_pre_df = pd.DataFrame(
    X_test_pre,
    columns=feature_names,
    index=X_test.index
)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_pre_df)

shap.summary_plot(shap_values, X_test_pre_df)
shap.summary_plot(shap_values, X_test_pre_df, plot_type='bar')


# SAVE MODEL
joblib.dump(best_model, 'model/best_model.pkl')
joblib.dump(X_test, 'model/x_test.pkl')
joblib.dump(y_test, 'model/y_test.pkl')
print('\nModel saved')