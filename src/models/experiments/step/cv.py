from models.experiments.base_step import ExperimentStep
from models.cross_validator import CrossValidator


class CrossValidationStep(ExperimentStep):

    def execute(self, context):

        for dataset_name, dataset in context.datasets.items():

            X = dataset["X"]
            y = dataset["y"]

            context.cv_results[dataset_name] = {}

            for model_type, model in context.models.items():

                cv = CrossValidator.stratified_kfold(
                    model,
                    X,
                    y
                )

                context.cv_results[dataset_name][model_type] = cv

                CrossValidator.print_kfold_results(
                    cv,
                    model.get_model_name()
                )