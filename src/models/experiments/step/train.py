from models.experiments.base_step import ExperimentStep

from models.factory import ModelFactory
from models.config import HYPERPARAMETER_SPACES


class TrainStep(ExperimentStep):

    def __init__(self, model_types):
        self.model_types = model_types

    def execute(self, context):

        first_dataset = next(iter(context.datasets.values()))

        X = first_dataset["X"]
        y = first_dataset["y"]

        for model_type in self.model_types:

            model = ModelFactory.create(
                model_type,
                HYPERPARAMETER_SPACES[model_type.value]["default"]
            )

            model.train(X, y)

            context.models[model_type] = model

            print(f"Trained {model.get_model_name()}")