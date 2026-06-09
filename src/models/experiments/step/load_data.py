from models.data_manager import DataManager
from models.experiments.base_step import ExperimentStep


class LoadDataStep(ExperimentStep):

    def __init__(self, datasets: list[str]):
        self.dataset_paths = datasets

    def execute(self, context):

        for path in self.dataset_paths:
            dm = DataManager(dataset_path=path)

            X, y = dm.load_data()

            context.datasets[path] = {
                "X": X,
                "y": y
            }

        print(f"Loaded {len(context.datasets)} datasets")