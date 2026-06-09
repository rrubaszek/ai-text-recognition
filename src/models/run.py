# from models.factory import ModelFactory
# from models.factory import ModelType
# from models.data_manager import DataManager
# from models.cross_validator import CrossValidator
# from models.config import HYPERPARAMETER_SPACES
# from models.tuning.hpo_optimizer import OptunaOptimizer
# from models.evaluation.lodo import LODOEvaluator
# from models.evaluation.statistics import StatisticalEvaluator
# from explainability.shap_analysis import shap_analysis_tree

# def _run_lodo_evaluation():
#     evaluator = LODOEvaluator()
#     evaluator.evaluate_all_models(verbose=True)
#     evaluator.summary()

# def _optimize_model(X, y, model_type: ModelType):
#     optimizer = OptunaOptimizer(model_type, n_trials=200)
#     _ = optimizer.optimize(X, y)
#     print(f"Best: {optimizer.get_best_params()}")

# def run(run_cv: bool = False, optimize: bool = False, lodo: bool = False, shap: bool = False):
#     full_data = DataManager(dataset_path="dataset.csv")
#     stylometry_data = DataManager(dataset_path="dataset_stylometry.csv")
    
#     X, y = full_data.load_data()
#     X_st, y_st = stylometry_data.load_data()
    
#     if lodo:
#         print("Running Leave-One-Domain-Out evaluation with lodo=True...")
#         _run_lodo_evaluation()
#         return
#     else:
#         print("Skipping Leave-One-Domain-Out evaluation for all models. To run LODO evaluation, use lodo=True.")

#     cv_results = []
#     cv_st_results = []
#     for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.LOGISTIC_REGRESSION, ModelType.RANDOM_FOREST]:
#         model = ModelFactory.create(model_type, HYPERPARAMETER_SPACES[model_type.value]['default'])
#         model.train(X, y)
#         if shap:
#             print(f"Running SHAP analysis for {model.get_model_name()} with shap=True...")
#             shap_analysis_tree(model, model_type.name, X)
#         else:
#             print(f"Skipping SHAP analysis for {model.get_model_name()}. To run SHAP analysis, use shap=True.")
        
#         if run_cv:
#             print(f"Running cross-validation for {model.get_model_name()} with run_cv=True...")
#             cv = CrossValidator.stratified_kfold(model, X, y)
#             cv_st = CrossValidator.stratified_kfold(model, X_st, y_st)
#             CrossValidator.print_kfold_results(cv, model.get_model_name())
#             CrossValidator.print_kfold_results(cv_st, model.get_model_name())
#             cv_results.append(cv)
#             cv_st_results.append(cv_st)
#         else:
#             print(f"Skipping cross-validation for {model.get_model_name()}. To run cross-validation, use run_cv=True.")
        
#         if optimize:
#             print(f"Running hyperparameter optimization for {model.get_model_name()} with optimize=True...")
#             _optimize_model(X, y, model_type)
#         else:
#             print(f"Skipping hyperparameter optimization for {model.get_model_name()}. To run optimization, use optimize=True.")

#     ev = StatisticalEvaluator(alpha=0.05, default_metric="test_roc_auc")

#     # cv_results from CrossValidator.stratified_kfold(...)
#     ev.compare_all(
#         cv_results,
#         model_names=[ModelType.XGBOOST.name, ModelType.LIGHTGBM.name, 
#                     ModelType.LOGISTIC_REGRESSION.name, ModelType.RANDOM_FOREST.name],
#         tests=("wilcoxon", "t_test"),   # pairwise
#         # Friedman + Nemenyi run automatically when ≥3 models
#     )
    
#     ev.compare_datasets(
#         cv_results,         
#         cv_st_results,
#         model_names=[ModelType.XGBOOST.name, ModelType.LIGHTGBM.name, 
#                     ModelType.LOGISTIC_REGRESSION.name, ModelType.RANDOM_FOREST.name],
#     )

# if __name__ == '__main__':
#     run(run_cv=True, optimize=False, lodo=False)

from models.experiments.builder import PipelineBuilder
from models.experiments.context import ExperimentContext
from models.experiments.config import ExperimentConfig

from models.factory import ModelType

def run():
    config = ExperimentConfig(
        datasets=[
            "dataset.csv",
            "dataset_stylometry.csv"
        ],

        models=[
            ModelType.XGBOOST,
            ModelType.LIGHTGBM,
            ModelType.LOGISTIC_REGRESSION,
            ModelType.RANDOM_FOREST
        ],

        cv=True,
        statistics=True,
        optimize=False,
        shap=False
    )

    builder = PipelineBuilder(config)

    pipeline = builder.build()
    
    pipeline.run(ExperimentContext())
    
if __name__ == "__main__":
    run()