import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, accuracy_score

from utils.paths import STYLOMETRY_DATASET_DIR
from sklearn.metrics import make_scorer, matthews_corrcoef


def main():
    df = pd.read_csv(STYLOMETRY_DATASET_DIR / "dataset.csv")
    # Zabezpieczenie wyrzucające metadane przed podaniem macierzy do modelu
    X = df.drop(columns=["label", "domain", "generator"], errors='ignore').values
    y = df["label"].astype(int).values

    # Random Forest jako baseline
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,  # Random Forest potrzebuje głębszych drzew niż XGBoost
        class_weight="balanced",  # Ważne dla stabilności
        random_state=42,
        n_jobs=-1
    )

    print("Rozpoczynam 5-krotną walidację krzyżową (5-Fold CV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mcc_scorer = make_scorer(matthews_corrcoef)

    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring={'accuracy': 'accuracy', 'f1_macro': 'f1_macro', 'roc_auc': 'roc_auc', 'mcc': mcc_scorer},
        return_train_score=False
    )

    print(f"\nŚrednie wyniki CV:")
    print(f"Accuracy: {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"F1-Macro: {np.mean(cv_results['test_f1_macro']):.4f}")
    print(f"ROC-AUC:  {np.mean(cv_results['test_roc_auc']):.4f}")
    print(f"MCC:      {np.mean(cv_results['test_mcc']):.4f}\n")

    # Trening na całości dla klasycznego raportu i przyszłej weryfikacji LODO/LOMO
    model.fit(X, y)
    print("Model bazowy został pomyślnie wytrenowany.")


if __name__ == "__main__":
    main()