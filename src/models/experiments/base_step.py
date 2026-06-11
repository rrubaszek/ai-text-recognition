from abc import ABC, abstractmethod


class ExperimentStep(ABC):

    @abstractmethod
    def execute(self, context):
        pass