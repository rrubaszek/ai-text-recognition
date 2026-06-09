class ExperimentPipeline:

    def __init__(self):
        self.steps = []

    def add(self, step):
        self.steps.append(step)

    def run(self, context):

        for step in self.steps:
            step.execute(context)