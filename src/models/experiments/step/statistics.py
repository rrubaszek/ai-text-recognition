from models.experiments.base_step import ExperimentStep

from models.evaluation.statistics import StatisticalEvaluator


class StatisticsStep(ExperimentStep):

    def execute(self, context):

        if len(context.datasets) < 2:
            print("Need at least 2 datasets")
            return

        dataset_names = list(context.datasets.keys())

        ds1 = dataset_names[0]
        ds2 = dataset_names[1]

        model_names = [
            model_type.name
            for model_type in context.models
        ]

        cv1 = [
            context.cv_results[ds1][m]
            for m in context.models
        ]

        cv2 = [
            context.cv_results[ds2][m]
            for m in context.models
        ]

        ev = StatisticalEvaluator(
            alpha=0.05,
            default_metric="test_roc_auc"
        )

        ev.compare_all(
            cv1,
            model_names=model_names
        )

        ev.compare_datasets(
            cv1,
            cv2,
            model_names=model_names
        )