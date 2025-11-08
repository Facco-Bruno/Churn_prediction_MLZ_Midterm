import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_STATE = 42


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.replace(' ', '_')

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df = df.dropna()

    df['Churn'] = (df['Churn'].str.strip() == 'Yes').astype(int)

    return df


def split_data(df: pd.DataFrame):
    df_full_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df['Churn']
    )

    df_train, df_val = train_test_split(
        df_full_train,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=df_full_train['Churn']
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def prepare_dicts(df: pd.DataFrame):
    cols_to_drop = [c for c in ['customerID', 'Churn'] if c in df.columns]
    df_prep = df.drop(columns=cols_to_drop, errors='ignore')
    return df_prep.to_dict(orient='records')


def train_logreg(dv, train_dicts, y_train):
    X_train = dv.fit_transform(train_dicts)

    Cs = [0.1, 1.0, 5.0, 10.0]
    best_auc = -1
    best_C = None
    best_model = None

    for C in Cs:
        model = LogisticRegression(C=C, max_iter=1000, n_jobs=-1)
        scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=5)
        mean_auc = scores.mean()
        print(f"LogReg C={C}: ROC AUC (CV) = {mean_auc:.3f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_C = C
            best_model = model

    print(f"Best LogReg C={best_C} ROC AUC (CV)={best_auc:.3f}")
    best_model.fit(X_train, y_train)
    return best_model, dv


def train_random_forest(dv, train_dicts, y_train):
    X_train = dv.fit_transform(train_dicts)

    n_estimators_options = [50, 100, 200]
    best_auc = -1
    best_n = None
    best_model = None

    for n in n_estimators_options:
        rf = RandomForestClassifier(
            n_estimators=n,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        scores = cross_val_score(rf, X_train, y_train, scoring='roc_auc', cv=5)
        mean_auc = scores.mean()
        print(f"RandomForest n_estimators={n}: ROC AUC (CV) = {mean_auc:.3f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_n = n
            best_model = rf

    print(f"Best RF n_estimators={best_n} ROC AUC (CV)={best_auc:.3f}")
    best_model.fit(X_train, y_train)
    return best_model, dv


def evaluate(model, dv, df_val):
    val_dicts = prepare_dicts(df_val)
    X_val = dv.transform(val_dicts)
    y_val = df_val['Churn'].values

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"ROC AUC (val) = {auc:.3f}")
    return auc


def main():
    print("Loading data...")
    df = load_data()

    print("Splitting train/val/test...")
    df_train, df_val, df_test = split_data(df)

    y_train = df_train['Churn'].values

    print("Preparing training dictionaries...")
    train_dicts = prepare_dicts(df_train)

    dv_logreg = DictVectorizer(sparse=True)
    dv_rf = DictVectorizer(sparse=True)

    print("\nTraining Logistic Regression...")
    logreg_model, dv_logreg = train_logreg(dv_logreg, train_dicts, y_train)

    print("\nTraining Random Forest...")
    rf_model, dv_rf = train_random_forest(dv_rf, train_dicts, y_train)

    print("\nEvaluating models on validation set...")
    auc_logreg = evaluate(logreg_model, dv_logreg, df_val)
    auc_rf = evaluate(rf_model, dv_rf, df_val)

    if auc_rf > auc_logreg:
        best_model = rf_model
        best_dv = dv_rf
        best_name = "RandomForest"
        best_auc = auc_rf
    else:
        best_model = logreg_model
        best_dv = dv_logreg
        best_name = "LogisticRegression"
        best_auc = auc_logreg

    print(f"\nSelected model: {best_name} (ROC AUC val = {best_auc:.3f})")

    print("\nEvaluating on test set...")
    test_dicts = prepare_dicts(df_test)
    X_test = best_dv.transform(test_dicts)
    y_test = df_test['Churn'].values
    y_test_pred = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred)
    print(f"ROC AUC (test) = {test_auc:.3f}")

    with open("model.bin", "wb") as f_out:
        pickle.dump(best_model, f_out)

    with open("dv.bin", "wb") as f_out:
        pickle.dump(best_dv, f_out)

    print("\nSaved model.bin and dv.bin")


if __name__ == "__main__":
    main()
