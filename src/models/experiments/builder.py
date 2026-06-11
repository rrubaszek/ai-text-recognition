
from models.experiments.pipeline import ExperimentPipeline

from models.experiments.step.load_data import LoadDataStep
from models.experiments.step.train import TrainStep
from models.experiments.step.cv import CrossValidationStep
from models.experiments.step.hpo import HyperparameterOptimizationStep
from models.experiments.step.shap import ShapStep
from models.experiments.step.statistics import StatisticsStep
from models.experiments.step.lodo import LODOStep


class PipelineBuilder:

    def __init__(self, config):
        self.config = config

    def build(self):

        pipeline = ExperimentPipeline()

        if self.config.lodo:
            pipeline.add(LODOStep())
            return pipeline

        pipeline.add(
            LoadDataStep(self.config.datasets)
        )

        if self.config.train:
            pipeline.add(
                TrainStep(self.config.models)
            )

        if self.config.cv:
            pipeline.add(
                CrossValidationStep()
            )

        if self.config.optimize:
            pipeline.add(
                HyperparameterOptimizationStep(
                    self.config.n_trials
                )
            )

        if self.config.shap:
            pipeline.add(
                ShapStep()
            )

        if self.config.statistics:
            pipeline.add(
                StatisticsStep()
            )

        return pipeline