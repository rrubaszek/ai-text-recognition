import csv

from preprocessing.tokenizer import SpacyTokenizer, SafeFeatures
from preprocessing.dynamics import NonlinearDynamics
from preprocessing.stylometry import ShallowStylometrics
from preprocessing.zipf import ZipfFeatures

from utils.load_data import load_data
from utils.paths import STYLOMETRY_DATASET_DIR

from sklearn.pipeline import FeatureUnion, Pipeline

def main():
    feature_pipeline = FeatureUnion([
        ("shallow", ShallowStylometrics()),
        ("zipf", ZipfFeatures()),
        ("nonlinear", NonlinearDynamics())
    ])
    
    pipeline = Pipeline([
        ("preprocess", SpacyTokenizer(use_lemma=False, remove_stopwords=False, remove_punct=True)),
        ("features", feature_pipeline)
    ])
    
    X = load_data()
    y = load_data(filename="labels.tsv")

    features = pipeline.fit_transform(X)
    print(features.shape)  # (n_samples, n_features)
    print(features)
    
    output_file = STYLOMETRY_DATASET_DIR / "dataset.csv"

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        n_features = features.shape[1]
        header = [f"f_{i}" for i in range(n_features)] + ["label"]
        writer.writerow(header)

        for vec, label in zip(features, y):
            row = list(vec) + [label]
            writer.writerow(row)

    print(f"Dataset saved to {output_file}")
    
if __name__ == '__main__':
    main()