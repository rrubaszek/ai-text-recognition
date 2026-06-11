from models.experiments.base_step import ExperimentStep

from explainability.shap_analysis import shap_analysis_tree


class ShapStep(ExperimentStep):

    def execute(self, context):

        first_dataset = next(iter(context.datasets.values()))

        X = first_dataset["X"]

        for model_type, model in context.models.items():

            shap_analysis_tree(
                model,
                model_type.name,
                X
            )