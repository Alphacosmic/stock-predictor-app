import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def train_model(ticker):
    df = pd.read_csv(f"data/{ticker}_features.csv")

    # ✅ Merge fundamentals
    fundamentals = pd.read_csv(f"data/{ticker}_fundamentals.csv")
    for col in fundamentals.columns:
        if col != "symbol":
            df[col] = fundamentals[col].iloc[0]
    df.dropna(inplace=True)

    #df = add_technical_indicators(df)
    df.dropna(inplace=True)

    df.to_csv(f"data/{ticker}_features.csv", index=False)
    print("[+] Saved enriched feature file")

    # Create binary target: 1 for up, -1 for down, skip 0 (flat)
    df["target"] = np.sign(df["Close"].shift(-1) - df["Close"])
    df.dropna(inplace=True)
    df = df[df['target'] != 0.0]

    # ✅ Drop non-numeric columns before modeling
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    df = df.drop(columns=non_numeric_cols)

    # Prepare features and labels
    X = df.drop(["Date", "target", "Close"], axis=1, errors='ignore')
    y = df["target"]

    # Drop classes with too few samples to avoid stratify error
    valid_classes = y.value_counts()[y.value_counts() >= 2].index
    X = X[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    # SMOTE oversampling
    sm = SMOTE(random_state=42, k_neighbors=1)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Convert labels: -1 → 0, 1 → 1
    y_train_res = y_train_res.map({-1: 0, 1: 1})
    y_test = y_test.map({-1: 0, 1: 1})

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train_res, y_train_res)

    # Save model
    with open(f"data/{ticker}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"[+] Trained and saved model to data/{ticker}_model.pkl")

    # Evaluate
    y_pred = model.predict(X_test)
    print("[✓] Model Evaluation:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model("AAPL")
