"""Reproducible training and evaluation for Disease_Presence."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "classification_data.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "disease_presence_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
TARGET = "Disease_Presence"
NUMERIC_FEATURES = ["Rainfall_mm", "Temperature_C", "Soil_Moisture", "Pest_Presence", "Day"]
CATEGORICAL_FEATURES = ["Region", "Crop_Type", "Fertilizer_Used"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_dataset() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    required = {"Date", TARGET, *CATEGORICAL_FEATURES, "Rainfall_mm",
                "Temperature_C", "Soil_Moisture", "Pest_Presence"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    dates = pd.to_datetime(data["Date"], format="%m/%d/%Y", errors="coerce")
    if dates.isna().any():
        raise ValueError("Date contains values that do not match MM/DD/YYYY.")
    data = data.copy()
    data["Day"] = dates.dt.day
    return data


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])


def main() -> None:
    data = load_dataset()
    labeled = data.loc[data[TARGET].notna()].copy()
    X = labeled[FEATURES]
    y = labeled[TARGET].astype(int)
    classes = set(y.unique())
    if classes != {0, 1}:
        raise ValueError(f"Expected binary target labels 0 and 1; found {sorted(classes)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "dataset_rows": int(len(data)),
        "labeled_rows": int(len(labeled)),
        "unlabeled_target_rows_excluded": int(data[TARGET].isna().sum()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target": TARGET,
        "target_classes": [0, 1],
        "target_interpretation": "Raw dataset class values; semantics are not independently documented.",
        "features": FEATURES,
        "split": {"test_size": 0.20, "random_state": 42, "stratify": True},
        "model": "LogisticRegression(class_weight='balanced', max_iter=2000)",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
        "confusion_matrix_labels_0_then_1": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1], output_dict=True, zero_division=0
        ),
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved pipeline: {MODEL_PATH}")
    print(f"Saved evaluation: {METRICS_PATH}")


if __name__ == "__main__":
    main()
