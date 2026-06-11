from models.experiments.builder import PipelineBuilder
from models.experiments.context import ExperimentContext
from models.experiments.config import ExperimentConfig

from models.factory import ModelType

def run():
    config = ExperimentConfig(
        datasets=[
            "dataset.csv",
            "dataset_stylometry.csv"
        ],

        models=[
            ModelType.XGBOOST,
            ModelType.LIGHTGBM,
            ModelType.LOGISTIC_REGRESSION,
            ModelType.RANDOM_FOREST
        ],

        train=True,
        cv=True,
        statistics=True,
        optimize=False,
        shap=False
    )

    builder = PipelineBuilder(config)

    pipeline = builder.build()
    
    pipeline.run(ExperimentContext())
    
if __name__ == "__main__":
    run()