import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float


def _infer_target_column(df: pd.DataFrame) -> str:
    """
    Tries common target column names. Falls back to last column if none match.
    """
    common_targets = ["target", "Target", "label", "Label", "y", "Y", "class", "Class", "output", "Output"]
    for c in common_targets:
        if c in df.columns:
            return c
    return df.columns[-1]


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def evaluate_models(
    df: pd.DataFrame,
    models: List[Tuple[str, object]],
    target_col: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_dir: str = "models",
) -> Dict:
    """
    Trains each model using a consistent preprocessing pipeline and returns metrics.

    Returns a dict that is JSON serializable.
    """
    if target_col is None:
        target_col = _infer_target_column(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # handle classification target if it is not numeric
    # (scikit models usually accept strings too, but keeping consistent)
    # We'll keep y as-is.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() <= 20 else None
    )

    preprocessor = _build_preprocessor(X)

    os.makedirs(save_dir, exist_ok=True)

    results: List[ModelResult] = []
    for name, model in models:
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # Binary metrics default: if multiclass, use macro averaging
        average = "binary" if pd.Series(y_test).nunique() == 2 else "macro"

        res = ModelResult(
            model_name=name,
            accuracy=float(accuracy_score(y_test, preds)),
            precision=float(precision_score(y_test, preds, average=average, zero_division=0)),
            recall=float(recall_score(y_test, preds, average=average, zero_division=0)),
            f1=float(f1_score(y_test, preds, average=average, zero_division=0)),
        )
        results.append(res)

        # Save trained pipeline for real-world use
        joblib.dump(pipeline, os.path.join(save_dir, f"{name}.joblib"))

    out = {
        "dataset": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "target_col": target_col,
        },
        "split": {"test_size": test_size, "random_state": random_state},
        "metrics": [
            {
                "model": r.model_name,
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
            }
            for r in results
        ],
    }
    return out


def save_metrics(metrics: Dict, filepath: str = "metrics.json") -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Dataset loaded but is empty.")
    return df

