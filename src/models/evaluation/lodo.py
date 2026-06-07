from typing import Dict, List, Any
import numpy as np
import pandas as pd

from models.factory import ModelFactory, ModelType
from models.data_manager import DataManager
from models.cross_validator import CrossValidator
from models.config import HYPERPARAMETER_SPACES


class LODOEvaluator:
    """Leave-One-Domain-Out evaluation framework"""
    
    def __init__(self, data_manager: DataManager = None):
        self.data_manager = data_manager or DataManager()
        self.results = {}
    
    def evaluate_model(
        self,
        model_type: ModelType,
        hyperparams: Dict[str, Any] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Evaluate a model using Leave-One-Domain-Out.
        """
        # Load data with domains
        X, y, domains = self.data_manager.load_with_domains()
        
        # Use default hyperparams if not provided
        if hyperparams is None:
            model_name = model_type.value
            if model_name not in HYPERPARAMETER_SPACES:
                raise ValueError(f"No hyperparams defined for {model_name}")
            hyperparams = HYPERPARAMETER_SPACES[model_name]['default']
        
        # Get model class from factory
        model_class = ModelFactory._registry[model_type]
        
        # Run LODO evaluation
        lodo_results = CrossValidator.leave_one_domain_out(
            model_class=model_class,
            X=X,
            y=y,
            domains=domains,
            hyperparams=hyperparams,
            verbose=verbose
        )
        
        self.results[model_type.value] = lodo_results
        
        return lodo_results
    
    def evaluate_all_models(
        self,
        hyperparams_dict: Dict[str, Dict[str, Any]] = None,
        verbose: bool = True
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Evaluate all available models using Leave-One-Domain-Out.
        """
        all_results = {}
        
        for model_type in ModelFactory._registry.keys():
            model_name = model_type.value
            
            # Get hyperparams for this model
            if hyperparams_dict and model_name in hyperparams_dict:
                hyperparams = hyperparams_dict[model_name]
            else:
                hyperparams = None
            
            print(f"\n{'─'*70}")
            print(f"Evaluating {model_name.upper()} with LODO...")
            print(f"{'─'*70}")
            
            # Evaluate
            lodo_results = self.evaluate_model(
                model_type,
                hyperparams=hyperparams,
                verbose=verbose
            )
            
            all_results[model_name] = lodo_results
        
        return all_results
    
    def compare_domains(self, metric: str = 'roc_auc') -> pd.DataFrame:
        """
        Compare model performance across domains.
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate_model() first.")
        
        if metric not in ['accuracy', 'f1_macro', 'roc_auc', 'mcc']:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Build comparison dataframe
        comparison_data = {}
        
        for model_name, lodo_result in self.results.items():
            domains = lodo_result['domain']
            scores = lodo_result[metric]
            
            for domain, score in zip(domains, scores):
                if domain not in comparison_data:
                    comparison_data[domain] = {}
                
                comparison_data[domain][model_name] = score
        
        df_comparison = pd.DataFrame(comparison_data).T
        
        # Add mean and std columns
        df_comparison['Mean'] = df_comparison.mean(axis=1)
        df_comparison['Std'] = df_comparison.std(axis=1)
        
        return df_comparison
    
    def summary(self) -> pd.DataFrame:
        if not self.results:
            raise ValueError("No results available. Run evaluate_model() first.")
        
        summary_data = []
        
        for model_name, lodo_result in self.results.items():
            summary_data.append({
                'Model': model_name.upper(),
                'Accuracy': f"{np.mean(lodo_result['accuracy']):.4f} (±{np.std(lodo_result['accuracy']):.4f})",
                'F1-Macro': f"{np.mean(lodo_result['f1_macro']):.4f} (±{np.std(lodo_result['f1_macro']):.4f})",
                'ROC-AUC': f"{np.mean(lodo_result['roc_auc']):.4f} (±{np.std(lodo_result['roc_auc']):.4f})",
                'MCC': f"{np.mean(lodo_result['mcc']):.4f} (±{np.std(lodo_result['mcc']):.4f})",
                'N_Domains': len(lodo_result['domain'])
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        print(f"\n{'='*100}")
        print(f"  Leave-One-Domain-Out (LODO) Summary")
        print(f"{'='*100}")
        print(df_summary.to_string(index=False))
        print(f"{'='*100}\n")
        
        return df_summary
