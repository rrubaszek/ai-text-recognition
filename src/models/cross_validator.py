from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    matthews_corrcoef
)
from models.base_model import BaseModel
from models.config import EVALUATION_METRICS


class CrossValidator:
    """Unified cross-validation framework"""
    
    DEFAULT_N_SPLITS = 5
    RANDOM_STATE = 42
    
    @staticmethod
    def stratified_kfold(
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        return_train_score: bool = False,
        verbose: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Perform K-Fold Stratified Cross-Validation.
        """
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=CrossValidator.RANDOM_STATE
        )
        
        scoring = {metric: EVALUATION_METRICS[metric] for metric in EVALUATION_METRICS}
        
        cv_results = cross_validate(
            model.model,
            X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=return_train_score,
            verbose=verbose
        )
        
        return cv_results
    
    @staticmethod
    def leave_one_domain_out(
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        domains: pd.Series,
        hyperparams: Dict[str, Any] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Perform Leave-One-Domain-Out evaluation.
        """
        unique_domains = sorted(domains.unique())
        results = {
            'domain': [],
            'accuracy': [],
            'f1_macro': [],
            'roc_auc': [],
            'mcc': [],
            'n_test_samples': []
        }
        
        for test_domain in unique_domains:
            # Split into train/test by domain
            test_mask = (domains == test_domain)
            train_mask = ~test_mask
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
    
            
            # Train model
            model = model_class(hyperparams=hyperparams)
            model.train(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            results['domain'].append(test_domain)
            results['accuracy'].append(accuracy_score(y_test, y_pred))
            results['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
            results['roc_auc'].append(roc_auc_score(y_test, y_proba))
            results['mcc'].append(matthews_corrcoef(y_test, y_pred))
            results['n_test_samples'].append(len(y_test))
        
        if verbose:
            CrossValidator.print_lodo_results(results)
        
        return results
    
    @staticmethod
    def print_kfold_results(
        cv_results: Dict[str, np.ndarray],
        model_name: str
    ) -> None:
        """Pretty-print K-Fold CV results"""
        print(f"\n{'='*60}")
        print(f"  {model_name} - 5-Fold Cross-Validation Results")
        print(f"{'='*60}")
        
        for metric in EVALUATION_METRICS:
            test_key = f'test_{metric}'
            if test_key in cv_results:
                scores = cv_results[test_key]
                print(f"{metric.upper():15} | Mean: {np.mean(scores):.4f} "
                      f"(±{np.std(scores):.4f})")
        
        print(f"{'='*60}\n")
    
    @staticmethod
    def print_lodo_results(lodo_results: Dict[str, List[float]]) -> None:
        """Pretty-print Leave-One-Domain-Out results"""
        df_results = pd.DataFrame(lodo_results)
        
        print(f"\n{'='*80}")
        print(f"  Leave-One-Domain-Out (LODO) Evaluation Results")
        print(f"{'='*80}")
        print(df_results.to_string(index=False))
        
        # Print summary
        for metric in EVALUATION_METRICS.keys: #['accuracy', 'f1_macro', 'roc_auc', 'mcc']:
            mean_val = np.mean(lodo_results[metric])
            std_val = np.std(lodo_results[metric])
            print(f"\n{metric.upper():15} | Mean: {mean_val:.4f} (±{std_val:.4f})")
        
        print(f"{'='*80}\n")