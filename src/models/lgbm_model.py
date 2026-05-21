import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef
import lightgbm as lgb
from utils.paths import STYLOMETRY_DATASET_DIR


def main():
    df = pd.read_csv(STYLOMETRY_DATASET_DIR / "dataset.csv")

    X = df.drop(columns=["label", "domain", "generator"], errors='ignore')
    y = df["label"].astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=1402,
        learning_rate=0.010366,
        num_leaves=196,
        max_depth=21,
        min_child_samples=27,
        subsample=0.509288,
        colsample_bytree=0.498542,
        reg_alpha=0.000000,
        reg_lambda=0.000040,
        #n_estimators=600,
        #learning_rate=0.03,
        #num_leaves=63,
        #max_depth=-1,
        #subsample=0.8,
        #colsample_bytree=0.8,
        #objective="binary",
        #random_state=42,
        #n_jobs=-1,
        #verbose=-1
    )

    print("Rozpoczynam 5-krotną walidację krzyżową (LightGBM)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Dodanie metryki MCC
    mcc_scorer = make_scorer(matthews_corrcoef)

    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring={'accuracy': 'accuracy', 'f1_macro': 'f1_macro', 'roc_auc': 'roc_auc', 'mcc': mcc_scorer},
        return_train_score=False
    )

    print(f"\nŚrednie wyniki LightGBM CV:")
    print(f"Accuracy: {np.mean(cv_results['test_accuracy']):.4f} (+/- {np.std(cv_results['test_accuracy']):.4f})")
    print(f"F1-Macro: {np.mean(cv_results['test_f1_macro']):.4f}")
    print(f"ROC-AUC:  {np.mean(cv_results['test_roc_auc']):.4f}")
    print(f"MCC:      {np.mean(cv_results['test_mcc']):.4f}\n")


if __name__ == "__main__":
    main()