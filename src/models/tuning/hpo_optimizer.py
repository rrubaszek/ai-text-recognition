from typing import Dict, Any, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler

from models.factory import ModelFactory, ModelType
from models.config import HYPERPARAMETER_SPACES, CV_CONFIG


class OptunaOptimizer:
    """Generic hyperparameter optimizer using Optuna and Bayesian optimization"""
    
    def __init__(
        self,
        model_type: ModelType,
        n_trials: int = 200,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1
    ):
        self.model_type = model_type
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.study = None
        self.best_params = None
        self.best_score = None
        self.model_name = model_type.value
        
        # Validate model type has search space defined
        if self.model_name not in HYPERPARAMETER_SPACES:
            raise ValueError(f"No search space defined for {self.model_name}")
        
        self.search_space = HYPERPARAMETER_SPACES[self.model_name].get('optuna_search_space', {})
        
        if not self.search_space:
            raise ValueError(f"Empty search space for {self.model_name}")
    
    def _build_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}
        
        for param_name, param_spec in self.search_space.items():
            if isinstance(param_spec, tuple):
                if len(param_spec) == 2:
                    # (min, max) - uniform or integer
                    min_val, max_val = param_spec
                    
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                    else:
                        params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                
                elif len(param_spec) == 3 and param_spec[2] == 'log':
                    # (min, max, 'log') - log scale
                    min_val, max_val = param_spec[0], param_spec[1]
                    params[param_name] = trial.suggest_float(
                        param_name, min_val, max_val, log=True
                    )
                else:
                    raise ValueError(f"Invalid search space spec: {param_spec}")
            
            elif isinstance(param_spec, list):
                # List of categorical values
                params[param_name] = trial.suggest_categorical(param_name, param_spec)
            
            else:
                raise ValueError(f"Unknown search space type for {param_name}: {param_spec}")
        
        return params
    
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        try:
            # Get hyperparameters from trial
            hyperparams = self._build_trial_params(trial)
            
            # Create model
            model = ModelFactory.create(self.model_type, hyperparams=hyperparams)
            
            # Get underlying sklearn model for cross_val_score
            underlying_model = model.model
            
            # Evaluate with cross-validation
            cv_scores = cross_val_score(
                underlying_model,
                X, y,
                cv=CV_CONFIG['n_splits'],
                scoring='roc_auc',
                n_jobs=self.n_jobs
            )
            
            # Return mean score
            return np.mean(cv_scores)
        
        except Exception as e:
            if self.verbose > 1:
                print(f"Trial failed with error: {e}")
            return -1.0  # Return worst possible score on error
    
    def optimize(self, X: np.ndarray, y: np.ndarray, show_progress: bool = True) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        """
        # Create sampler with random state for reproducibility
        sampler = TPESampler(seed=self.random_state)
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=show_progress,
            n_jobs=self.n_jobs
        )
        
        # Get best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"  Optimization Complete: {self.model_name.upper()}")
            print(f"{'='*70}")
            print(f"Number of trials: {len(self.study.trials)}")
            print(f"Best ROC-AUC: {self.best_score:.4f}")
            print(f"\nBest hyperparameters:")
            for param, value in self.best_params.items():
                print(f"  {param:25} = {value}")
            print(f"{'='*70}\n")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': self.study
        }
    
    def get_best_params(self) -> Dict[str, Any]:
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
        return self.best_params.copy()
    
    def get_best_score(self) -> float:
        if self.best_score is None:
            raise ValueError("Must run optimize() first")
        return self.best_score
    
    def get_trials_dataframe(self) -> pd.DataFrame:
        if self.study is None:
            raise ValueError("Must run optimize() first")
        return self.study.trials_dataframe()
