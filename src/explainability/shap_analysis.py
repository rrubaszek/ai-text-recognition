import matplotlib.pyplot as plt
import shap

from utils.paths import SHAP_PLOTS_DIR

def shap_analysis_tree(model, model_name, x_train):
    # Analiza SHAP
    plot_dir = SHAP_PLOTS_DIR / model_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_dir / "shap_values_mean.png"

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x_train)

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
    shap.plots.scatter(shap_values[:, "Comma_Ratio"], color=shap_values[:, "Dot_Ratio"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_comma_ratio_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:, "Var_Sent_Len"], color=shap_values[:, "Avg_Sent_Len"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_var_sent_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:, "Dot_Ratio"], color=shap_values[:, "Avg_Sent_Len"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_dot_ratio_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:, "Avg_Sent_Len"], color=shap_values[:, "Comma_Ratio"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_avg_sent_len_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:, "Var_Word_Len"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_var_word_len_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:, "Avg_Dep_Depth"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_avg_dep_depth_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:, "Metric_R"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_metric_r_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:, "Yules_K"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_yules_k_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.scatter(shap_values[:, "Entropy"], show=False)
    plt.gcf().subplots_adjust(left=0.55)
    plt.tight_layout()
    plot_path = plot_dir / "shap_values_entropy_dependence.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()