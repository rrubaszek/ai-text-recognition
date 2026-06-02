import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from utils.paths import STYLOMETRY_DATASET_DIR

def main():
    # 1. Wczytanie danych
    df = pd.read_csv(STYLOMETRY_DATASET_DIR / "dataset.csv")

    # Wyrzucamy wszystkie metadane z macierzy X
    X = df.drop(columns=["label", "domain", "generator"], errors='ignore')
    y = df["label"].astype(int).values

    feature_names = [
        # Pillar 1: Shallow Stylometrics (17 cech)
        "Token_Count", "Char_Length", "Comma_Ratio", "Dot_Ratio", "Avg_Sent_Len", "Var_Sent_Len",
        "TTR", "HLR", "Entropy", "Avg_Word_Len", "Var_Word_Len", "Yules_K",
        "Noun_Ratio", "Verb_Ratio", "Adj_Ratio", "Avg_Dep_Depth", "Max_Dep_Depth",

        # Pillar 2: Macroscopic Frequency Errors (6 cech)
        "Zipf_MAPE", "Zipf_Correlation", "Zipf_Slope", "Zipf_Residual_Var", "Zipf_KS_Stat", "Metric_R",

        # Pillar 3: Nonlinear Dynamics (14 cech)
        "Taylor_b",
        "RQA_Recurrence_Rate", "RQA_Determinism", "RQA_Entropy", "RQA_Laminarity",
        "CRQA_Recurrence_Rate", "CRQA_Determinism", "CRQA_Entropy", "CRQA_Laminarity",
        "NVG_Avg_Degree", "NVG_Clustering", "NVG_Avg_Path_Length", "NVG_Graph_Entropy",
        "MSE_Entropy_Variance"
    ]

    # Zabezpieczenie przed niezgodnością wymiarów
    if X.shape[1] == len(feature_names):
        X.columns = feature_names
    else:
        print(f"KRYTYCZNY BŁĄD: Oczekiwano {len(feature_names)} cech, ale model ma {X.shape[1]}.")
        return

    # 3. Trening modelu bazowego do wyjaśnienia
    print("Trenowanie modelu Random Forest do analizy SHAP...")
    # POPRAWKA 3: Ujednolicenie max_depth z plikiem train.py
    model = RandomForestClassifier(n_estimators=400, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X, y)

    # 4. Wyliczenie wartości SHAP
    print("Obliczanie wartości SHAP (to zajmie chwilę)...")
    explainer = shap.TreeExplainer(model)
    # Wyłączenie check_additivity bywa pomocne przy głębokich drzewach
    shap_values = explainer.shap_values(X, check_additivity=False)

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values

    # 5. Wygenerowanie i zapisanie wykresu
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_to_plot, X, show=False, max_display=15)

    plot_path = STYLOMETRY_DATASET_DIR / "shap_summary.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"\nWykres SHAP zapisany do: {plot_path}")

if __name__ == "__main__":
    main()