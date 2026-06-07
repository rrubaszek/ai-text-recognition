from models.factory import ModelFactory
from models.factory import ModelType
from models.data_manager import DataManager
from models.cross_validator import CrossValidator
from models.config import HYPERPARAMETER_SPACES
from models.tuning.hpo_optimizer import OptunaOptimizer
from models.evaluation.lodo import LODOEvaluator
from explainability.shap_analysis import shap_analysis_tree

def _run_lodo_evaluation():
    evaluator = LODOEvaluator()
    evaluator.evaluate_all_models(verbose=True)
    evaluator.summary()

def _optimize_model(X, y, model_type: ModelType):
    optimizer = OptunaOptimizer(model_type, n_trials=200)
    _ = optimizer.optimize(X, y)
    print(f"Best: {optimizer.get_best_params()}")

def run(run_cv: bool = False, optimize: bool = False, lodo: bool = False, shap: bool = False, stylometric_only: bool = False, lemmatization_used: bool = True):
    dm = DataManager(lemmatization_used= lemmatization_used)
    X, y = dm.load_data()
    print(f"Loaded {X.shape[0]} samples and {X.shape[1]} features.")

    if stylometric_only:
        # First 18 features are stylometric features
        X = X.iloc[:, :18]

    if lodo:
        print("Running Leave-One-Domain-Out evaluation with lodo=True...")
        _run_lodo_evaluation()
        return
    else:
        print("Skipping Leave-One-Domain-Out evaluation for all models. To run LODO evaluation, use lodo=True.")

    for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.LOGISTIC_REGRESSION, ModelType.RANDOM_FOREST]:
        model = ModelFactory.create(model_type, HYPERPARAMETER_SPACES[model_type.value]['default'])
        model.train(X, y)
        if shap:
            print(f"Running SHAP analysis for {model.get_model_name()} with shap=True...")
            shap_analysis_tree(model, model_type.name, X, lemmatization_used)
        else:
            print(f"Skipping SHAP analysis for {model.get_model_name()}. To run SHAP analysis, use shap=True.")

        if run_cv:
            print(f"Running cross-validation for {model.get_model_name()} with run_cv=True...")
            cv_results = CrossValidator.stratified_kfold(model, X, y)
            CrossValidator.print_kfold_results(cv_results, model.get_model_name())
        else:
            print(f"Skipping cross-validation for {model.get_model_name()}. To run cross-validation, use run_cv=True.")

        if optimize:
            print(f"Running hyperparameter optimization for {model.get_model_name()} with optimize=True...")
            _optimize_model(X, y, model_type)
        else:
            print(f"Skipping hyperparameter optimization for {model.get_model_name()}. To run optimization, use optimize=True.")


if __name__ == '__main__':
    run(run_cv=True, optimize=False, lodo=False, stylometric_only=False, shap=True)