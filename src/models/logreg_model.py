import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils.paths import STYLOMETRY_DATASET_DIR

def main():
    df = pd.read_csv(STYLOMETRY_DATASET_DIR / "dataset.csv")

    X = df.drop(columns=["label", "domain", "generator"], errors='ignore').values
    y = df["label"].astype(int).values

    # UWAGA: Regresja Logistyczna wymaga standaryzacji (skalowania) cech
    # w przeciwnym razie wagi wybuchną przez metryki takie jak Token_Count
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])

    print("Rozpoczynam 5-krotną walidację krzyżową (Logistic Regression - Baseline)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mcc_scorer = make_scorer(matthews_corrcoef)

    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring={'accuracy': 'accuracy', 'f1_macro': 'f1_macro', 'roc_auc': 'roc_auc', 'mcc': mcc_scorer},
        return_train_score=False
    )

    print(f"\nŚrednie wyniki Logistic Regression CV:")
    print(f"Accuracy: {np.mean(cv_results['test_accuracy']):.4f} (+/- {np.std(cv_results['test_accuracy']):.4f})")
    print(f"F1-Macro: {np.mean(cv_results['test_f1_macro']):.4f}")
    print(f"ROC-AUC:  {np.mean(cv_results['test_roc_auc']):.4f}")
    print(f"MCC:      {np.mean(cv_results['test_mcc']):.4f}\n")

if __name__ == "__main__":
    main()