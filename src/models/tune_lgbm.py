import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings

# Wyłączamy ostrzeżenia, aby logi Optuny były czytelne
warnings.filterwarnings("ignore")

from utils.paths import STYLOMETRY_DATASET_DIR


def objective(trial):
    # 1. Wczytanie danych (wewnątrz funkcji, by zapewnić czystość pamięci)
    df = pd.read_csv(STYLOMETRY_DATASET_DIR / "dataset.csv")
    X = df.drop(columns=["label", "domain", "generator"], errors='ignore').values
    y = df["label"].astype(int).values

    # 2. Przestrzeń poszukiwań (Search Space) dla LightGBM
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,

        # Kluczowe hiperparametry do optymalizacji
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),

        # Regularyzacja (L1 i L2) - zabezpiecza przed przeuczeniem
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    # 3. Inicjalizacja modelu
    model = lgb.LGBMClassifier(**param)

    # 4. Walidacja krzyżowa (5-Fold CV)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Optymalizujemy pod kątem ROC-AUC, bo to kluczowa metryka w Twoim artykule
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    return np.mean(scores)


def main():
    print("Rozpoczynam Optymalizację Bayesowską (Optuna) dla LightGBM...")
    print("Algorytm przetestuje 200 kombinacji parametrów (1000 treningów przez 5-Fold CV).")
    print("Zaparz kawę, to potrwa od kilkunastu do kilkudziesięciu minut...\n")

    # Tworzymy badanie maksymalizujące wynik (bo im większe ROC-AUC, tym lepiej)
    study = optuna.create_study(direction='maximize', study_name="LGBM_Stylometry_Optimization")

    # Uruchamiamy 200 iteracji
    study.optimize(objective, n_trials=200)

    # Podsumowanie wyników
    print("\n" + "=" * 50)
    print("🎯 KONIEC OPTYMALIZACJI - ZNALEZIONO ŚWIĘTEgo GRAALA 🎯")
    print("=" * 50)
    print(f"🏆 Najlepszy średni wynik ROC-AUC: {study.best_value:.4f}")
    print("\nSkopiuj poniższe parametry i wklej je do pliku lgbm_model.py:")
    print("-" * 30)

    for key, value in study.best_params.items():
        # Formatowanie odpowiednie do wklejenia w Pythonie
        if isinstance(value, float):
            print(f"        {key}={value:.6f},")
        else:
            print(f"        {key}={value},")

    print("-" * 30)


if __name__ == "__main__":
    main()