from dataclasses import dataclass, field


@dataclass
class ExperimentContext:
    datasets: dict = field(default_factory=dict)

    models: dict = field(default_factory=dict)

    cv_results: dict = field(default_factory=dict)

    optimized_params: dict = field(default_factory=dict)