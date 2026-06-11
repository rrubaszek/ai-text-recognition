from models.experiments.base_step import ExperimentStep

from models.evaluation.lodo import LODOEvaluator


class LODOStep(ExperimentStep):

    def execute(self, context):

        evaluator = LODOEvaluator()

        evaluator.evaluate_all_models(
            verbose=True
        )

        evaluator.summary()