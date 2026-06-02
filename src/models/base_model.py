from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all classification models"""
    
    def __init__(self, hyperparams: Dict[str, Any] = None, random_state: int = 42):
        self.hyperparams = hyperparams or {}
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def _build_model(self):
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass
    
    def predict_proba(self, X) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support predict_proba"
        )
    
    @abstractmethod
    def get_model_name(self) -> str:
        pass
    
    def get_hyperparams(self) -> Dict[str, Any]:
        return self.hyperparams.copy()
    
    def update_hyperparams(self, **kwargs):
        """(should be called before training)"""
        self.hyperparams.update(kwargs)
        if self.model is not None:
            self._build_model()
    
    def __repr__(self) -> str:
        return f"{self.get_model_name()}(hyperparams={self.hyperparams})"