from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np


class DataManager:
    """
    Data loader for preprocessed stylometric features.
    
    Feature extraction happens in src/preprocessing/ pipeline (main.py)
    This class only loads the already-processed CSV output.
    """
    
    def __init__(self, dataset_path: Path = None):
        if dataset_path is None:
            from utils.paths import STYLOMETRY_DATASET_DIR
            dataset_path = STYLOMETRY_DATASET_DIR / "dataset.csv"
        
        self.dataset_path = Path(dataset_path)
        self._data = None
        self._X = None
        self._y = None
        self._domains = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load preprocessed data (X features and y labels).
        
        Returns:
            Tuple of (X_features, y_labels) where X_features includes only the feature columns
        """
        self._data = pd.read_csv(self.dataset_path)
        
        # Extract features (drop label and metadata columns)
        self._X = self._data.drop(
            columns=["label", "domain", "generator"],
            errors='ignore'
        )
        self._y = self._data["label"].astype(int)
        
        return self._X, self._y
    
    def load_with_domains(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        if self._data is None:
            self._data = pd.read_csv(self.dataset_path)
        
        self._X = self._data.drop(
            columns=["label", "domain", "generator"],
            errors='ignore'
        )
        self._y = self._data["label"].astype(int)
        self._domains = self._data.get("domain", pd.Series(["unknown"] * len(self._data)))
        
        return self._X, self._y, self._domains
    
    def get_feature_names(self) -> list:
        if self._X is None:
            self.load_data()
        return self._X.columns.tolist()
    
    def get_data_as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data as NumPy arrays (compatible with sklearn models).
        """
        if self._X is None:
            self.load_data()
        
        return self._X.values, self._y.values
    
    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = pd.read_csv(self.dataset_path)
        return self._data
    
    @property
    def X(self) -> pd.DataFrame:
        if self._X is None:
            self.load_data()
        return self._X
    
    @property
    def y(self) -> pd.Series:
        if self._y is None:
            self.load_data()
        return self._y
    
    @property
    def domains(self) -> pd.Series:
        if self._domains is None:
            self.load_with_domains()
        return self._domains