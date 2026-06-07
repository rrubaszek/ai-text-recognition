from typing import Dict, Any, Type
from enum import Enum
from models.base_model import BaseModel
from models.implementations.xgb_model import XGBClassifierModel
from models.implementations.lgbm_model import LGBMClassifierModel
from models.implementations.logreg_model import LogisticRegressionModel
from models.implementations.randfor_model import RandomForestModel


class ModelType(Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LOGISTIC_REGRESSION = "logreg"
    RANDOM_FOREST = "randfor"


class ModelFactory:
    _registry: Dict[ModelType, Type[BaseModel]] = {
        ModelType.XGBOOST: XGBClassifierModel,
        ModelType.LIGHTGBM: LGBMClassifierModel,
        ModelType.LOGISTIC_REGRESSION: LogisticRegressionModel,
        ModelType.RANDOM_FOREST: RandomForestModel,
    }
    
    @classmethod
    def create(cls, model_type: ModelType, hyperparams: Dict[str, Any] = None, **kwargs) -> BaseModel:
        if model_type not in cls._registry:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(cls._registry.keys())}"
            )
        
        model_class = cls._registry[model_type]
        return model_class(hyperparams=hyperparams, **kwargs)
    
    @classmethod
    def create_from_string(cls, model_name: str, hyperparams: Dict = None) -> BaseModel:
        try:
            model_type = ModelType(model_name.lower())
            return cls.create(model_type, hyperparams)
        except ValueError:
            raise ValueError(f"Unknown model name: {model_name}")
    
        
    @classmethod
    def list_available_models(cls) -> list:
        return [m.value for m in ModelType]
    