import csv
import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline
from preprocessing.tokenizer import SpacyTokenizer, SafeFeatures
from preprocessing.dynamics import NonlinearDynamics
from preprocessing.stylometry import ShallowStylometrics
from preprocessing.zipf import ZipfFeatures
from utils.load_data import load_data
from utils.paths import STYLOMETRY_DATASET_DIR

def run(force: bool = False):
    
    output_file = STYLOMETRY_DATASET_DIR / "dataset.csv"
    STYLOMETRY_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    if output_file.exists() and not force:
        print(f"Output file {output_file} already exists. Use force=True to overwrite.")
        return

    feature_pipeline = FeatureUnion([
        ("shallow", ShallowStylometrics()),
        ("zipf", ZipfFeatures()),
        ("nonlinear", NonlinearDynamics())
    ])

    # dla zgodności z artykułem pl_core_news_lg oraz włączamy lematyzację
    pipeline = Pipeline([
        ("preprocess", SpacyTokenizer(
            model_name="pl_core_news_lg",
            use_lemma=True,
            remove_stopwords=False,
            remove_punct=False
        )),
        ("features", feature_pipeline),
        ("safe_cast", SafeFeatures())
    ])

    print("Ładowanie danych...")
    X = load_data(filename="data.tsv", type="train", is_labels=False)
    y = load_data(filename="labels.tsv", type="train", is_labels=True)

    print("Ekstrakcja cech fizycznych...")
    features = pipeline.fit_transform(X)

    # Use predefined feature list from config to ensure correct column order in output CSV
    from models.config import STYLOMETRIC_FEATURES
    # lista cech
    # feature_names = [
    #     # Pillar 1: Shallow Stylometrics (17 cech)
    #     "Token_Count", "Char_Length", "Comma_Ratio", "Dot_Ratio", "Avg_Sent_Len", "Var_Sent_Len",
    #     "TTR", "HLR", "Entropy", "Avg_Word_Len", "Var_Word_Len", "Yules_K",
    #     "Noun_Ratio", "Verb_Ratio", "Adj_Ratio", "Avg_Dep_Depth", "Max_Dep_Depth",

    #     # Pillar 2: Macroscopic Frequency Errors (6 cech)
    #     "Zipf_MAPE", "Zipf_Correlation", "Zipf_Slope", "Zipf_Residual_Var", "Zipf_KS_Stat", "Metric_R",

    #     # Pillar 3: Nonlinear Dynamics (14 cech)
    #     "Taylor_b",
    #     "RQA_Recurrence_Rate", "RQA_Determinism", "RQA_Entropy", "RQA_Laminarity",
    #     "CRQA_Recurrence_Rate", "CRQA_Determinism", "CRQA_Entropy", "CRQA_Laminarity",
    #     "NVG_Avg_Degree", "NVG_Clustering", "NVG_Avg_Path_Length", "NVG_Graph_Entropy",
    #     "MSE_Entropy_Variance"
    # ]

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = STYLOMETRIC_FEATURES + ["label", "domain", "generator"]
        writer.writerow(header)

        # UWAGA: W przyszłości trzeba zmodyfikować load_data by zwracało domenę i generator
        # Na ten moment chronimy pipeline wstawiając placeholdery.
        for vec, label in zip(features, y):
            writer.writerow(list(vec) + [label, "unknown", "unknown"])

if __name__ == '__main__':
    run(force=False)