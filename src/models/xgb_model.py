import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

from utils.paths import STYLOMETRY_DATASET_DIR

def main():
    df = pd.read_csv(STYLOMETRY_DATASET_DIR / "dataset.csv")
    # Zabezpieczenie wyrzucające metadane przed podaniem macierzy do modelu
    X = df.drop(columns=["label", "domain", "generator"], errors='ignore').values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=1292,
        learning_rate=0.011540,
        max_depth=12,
        min_child_weight=5,
        subsample=0.586369,
        colsample_bytree=0.889294,
        gamma=1.305129,
        reg_lambda=3.997312,
        #n_estimators=400,
        #max_depth=7,  # Średnia głębokość do wyłapania interakcji nieliniowych
        #learning_rate=0.03,  # Wolniejsza, ale precyzyjniejsza nauka
        #subsample=0.8,  # Zabezpieczenie przed przeuczeniem
        #colsample_bytree=0.9,    # Używamy prawie wszystkich cech w każdym drzewie (mamy ich tylko 37)
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()