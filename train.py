import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from model_utils import evaluate_models, save_metrics, load_dataset

if __name__ == "__main__":
    df = load_dataset("data/heart.csv")

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=2000)),
        ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
        ("SVM_RBF", SVC(kernel="rbf", probability=True, random_state=42)),
    ]

    metrics = evaluate_models(df, models=models, target_col=None)
    save_metrics(metrics, "metrics.json")
    print("Saved metrics.json")