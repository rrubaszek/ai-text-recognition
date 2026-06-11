from models.experiments.base_step import ExperimentStep

from models.tuning.hpo_optimizer import OptunaOptimizer

class HyperparameterOptimizationStep(ExperimentStep):

    def __init__(self, n_trials: int):
        self.n_trials = n_trials

    def execute(self, context):

        first_dataset = next(iter(context.datasets.values()))

        X = first_dataset["X"]
        y = first_dataset["y"]

        for model_type in context.models:

            optimizer = OptunaOptimizer(
                model_type,
                n_trials=self.n_trials
            )

            optimizer.optimize(X, y)

            context.optimized_params[model_type] = (
                optimizer.get_best_params()
            )

            print(
                f"Best params for {model_type.name}: "
                f"{optimizer.get_best_params()}"
            )