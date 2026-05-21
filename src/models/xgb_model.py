from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import shap

from utils.paths import STYLOMETRY_DATASET_DIR

def main():
    df = pd.read_csv(STYLOMETRY_DATASET_DIR / "dataset.csv")
    # Zabezpieczenie wyrzucające metadane przed podaniem macierzy do modelu
    X = df.drop(columns=["label", "domain", "generator"], errors='ignore')
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=7,  # Średnia głębokość do wyłapania interakcji nieliniowych
        learning_rate=0.03,  # Wolniejsza, ale precyzyjniejsza nauka
        subsample=0.8,  # Zabezpieczenie przed przeuczeniem
        colsample_bytree=0.9,    # Używamy prawie wszystkich cech w każdym drzewie (mamy ich tylko 37)
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

    # Analiza SHAP
    project_root = Path(__file__).resolve().parents[2]
    plot_dir = project_root / "src" / "plots" / "xgb_model"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_dir / "shap_values_mean.png"

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)

    plt.figure(figsize=(12, 6))
    shap.plots.bar(shap_values, show=False)
    plt.gcf().subplots_adjust(left=0.35)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_beeswarm.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:,"Comma_Ratio"], color=shap_values[:,"Dot_Ratio"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_comma_ratio_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:,"Var_Sent_Len"], color=shap_values[:,"Avg_Sent_Len"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_var_sent_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:,"Dot_Ratio"], color=shap_values[:,"Avg_Sent_Len"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_dot_ratio_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:,"Avg_Sent_Len"], color=shap_values[:,"Comma_Ratio"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_avg_sent_len_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:,"Var_Word_Len"],  show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_var_word_len_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:,"Avg_Dep_Depth"],  show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_avg_dep_depth_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:,"Metric_R"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_metric_r_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:,"Yules_K"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_yules_k_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:,"Entropy"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_entropy_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()