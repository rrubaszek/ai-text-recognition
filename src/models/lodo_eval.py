import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import lightgbm as lgb
from utils.paths import STYLOMETRY_DATASET_DIR
import warnings

warnings.filterwarnings("ignore")


def main():
    print("Ładowanie macierzy cech...")
    df = pd.read_csv(STYLOMETRY_DATASET_DIR / "dataset.csv")

    # 1. DYNAMICZNE ETYKIETOWANIE DOMEN (Kwantylowe / Tertyle)
    # Ze względu na to, że organizatorzy sztucznie ucięli teksty (truncation limit),
    # używamy percentyli do równego podziału zbioru na 3 domeny proxy.

    if df['domain'].nunique() <= 1 or df['domain'].iloc[0] == 'unknown':
        print("Wykryto brak metadanych. Wykonuję równy, kwantylowy podział na domeny proxy...")
        # rank(method='first') upewnia się, że nawet przy duplikatach długości podział będzie równy
        df['domain'] = pd.qcut(
            df['Char_Length'].rank(method='first'),
            q=3,
            labels=['domain_short', 'domain_medium', 'domain_long']
        ).astype(str)

    domains = df['domain'].unique()
    print(f"\nZidentyfikowane domeny w zbiorze: {list(domains)}")

    feature_cols = [c for c in df.columns if c not in ["label", "domain", "generator"]]

    print("\n" + "=" * 60)
    print("EKSPERYMENT LEAVE-ONE-DOMAIN-OUT (LODO)")
    print("=" * 60)

    lodo_results = []

    for test_domain in domains:
        # Zgodnie z metodologią LODO: Trenujemy na wszystkim OPRÓCZ jednej domeny
        train_df = df[df['domain'] != test_domain]
        # Testujemy WYŁĄCZNIE na odizolowanej domenie
        test_df = df[df['domain'] == test_domain]

        X_train, y_train = train_df[feature_cols].values, train_df['label'].astype(int).values
        X_test, y_test = test_df[feature_cols].values, test_df['label'].astype(int).values

        # Zoptymalizowane parametry LightGBM dla cech fizycznych
        model = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds, average='macro')

        print(f" DOMENA TESTOWA: {test_domain.upper()}")
        print(f" Próbki Treningowe: {len(train_df)} | Próbki Testowe: {len(test_df)}")
        print(f" Wynik -> Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f} | F1: {f1:.4f}\n")

        lodo_results.append((test_domain, acc, auc))

    # Podsumowanie naukowe
    print("=" * 60)
    print("📊 WNIOSKI Z EKSPERYMENTU LODO:")
    print("=" * 60)

    # Sortowanie po wynikach ROC-AUC
    lodo_results.sort(key=lambda x: x[2], reverse=True)
    for res in lodo_results:
        print(f"Domena: {res[0]:<25} | Osiągnięte ROC-AUC: {res[2]:.4f}")


if __name__ == "__main__":
    main()