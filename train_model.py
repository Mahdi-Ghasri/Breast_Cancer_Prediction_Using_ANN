
import json
import os
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def train_and_save(artifacts_dir: str = "model") -> None:
    output_dir = Path(__file__).resolve().parent / artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)


    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names.tolist()

    # Define a simple pipeline: scaling + logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=10_000)),
    ])

    # Fit the pipeline
    pipeline.fit(X, y)

    # Save the trained pipeline
    model_path = output_dir / "model.pkl"
    joblib.dump(pipeline, model_path)

    # Save the feature names
    feats_path = output_dir / feature_names
    with open(feats_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved feature names to {feats_path}")


if __name__ == "__main__":
    train_and_save()