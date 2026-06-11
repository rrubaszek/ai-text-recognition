from dataclasses import dataclass, field
from models.factory import ModelType


@dataclass
class ExperimentConfig:
    datasets: list[str]

    models: list[ModelType] = field(default_factory=lambda: [
        ModelType.XGBOOST,
        ModelType.LIGHTGBM,
        ModelType.LOGISTIC_REGRESSION,
        ModelType.RANDOM_FOREST
    ])

    train: bool = True
    cv: bool = False
    optimize: bool = False
    shap: bool = False
    statistics: bool = False
    lodo: bool = False

    n_trials: int = 200