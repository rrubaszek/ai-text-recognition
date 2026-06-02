from sklearn.metrics import (
    accuracy_score, f1_score, make_scorer, roc_auc_score, matthews_corrcoef
)

# Evaluation metrics used across all models
EVALUATION_METRICS = {
    'accuracy': make_scorer(accuracy_score),
    'f1_macro': make_scorer(f1_score, average='macro'),
    'mcc': make_scorer(matthews_corrcoef),
    'roc_auc': 'roc_auc',
}

# Feature names (from src/main.py preprocessing pipeline)
STYLOMETRIC_FEATURES = [
    # Pillar 1: Shallow Stylometrics (17 features)
    "Token_Count", "Char_Length", "Comma_Ratio", "Dot_Ratio", "Avg_Sent_Len", "Var_Sent_Len",
    "TTR", "HLR", "Entropy", "Avg_Word_Len", "Var_Word_Len", "Yules_K",
    "Noun_Ratio", "Verb_Ratio", "Adj_Ratio", "Avg_Dep_Depth", "Max_Dep_Depth",

    # Pillar 2: Macroscopic Frequency Errors (6 features)
    "Zipf_MAPE", "Zipf_Correlation", "Zipf_Slope", "Zipf_Residual_Var", "Zipf_KS_Stat", "Metric_R",

    # Pillar 3: Nonlinear Dynamics (14 features)
    "Taylor_b",
    "RQA_Recurrence_Rate", "RQA_Determinism", "RQA_Entropy", "RQA_Laminarity",
    "CRQA_Recurrence_Rate", "CRQA_Determinism", "CRQA_Entropy", "CRQA_Laminarity",
    "NVG_Avg_Degree", "NVG_Clustering", "NVG_Avg_Path_Length", "NVG_Graph_Entropy",
    "MSE_Entropy_Variance"
]

# Hyperparameter configurations
HYPERPARAMETER_SPACES = {
    'xgboost': {
        'default': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_lambda': 1,
            'reg_alpha': 0,
            'random_state': 42,
        },
        'optuna_search_space': {
            'n_estimators': (200, 1500),
            'learning_rate': (0.005, 0.1, 'log'),
            'max_depth': (4, 15),
            'min_child_weight': (1, 20),
            'subsample': (0.4, 1.0),
            'colsample_bytree': (0.4, 1.0),
            'gamma': (0, 10),
            'reg_lambda': (1e-8, 10, 'log'),
        }
    },
    'lightgbm': {
        'default': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'random_state': 42,
        },
        'optuna_search_space': {
            'n_estimators': (200, 1500),
            'learning_rate': (0.005, 0.1, 'log'),
            'num_leaves': (15, 255),
            'max_depth': (5, 25),
            'min_child_samples': (5, 100),
            'subsample': (0.4, 1.0),
            'colsample_bytree': (0.4, 1.0),
            'reg_alpha': (1e-8, 10, 'log'),
            'reg_lambda': (1e-8, 10, 'log'),
        }
    },
    'logreg': {
        'default': {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42,
        },
        'optuna_search_space': {
            'C': (1e-4, 100, 'log'),
            'penalty': ['l2', 'l1'],
        }
    },
    'randfor': {
        'default': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
        },
        'optuna_search_space': {
            'n_estimators': (50, 500),
            'max_depth': (5, 50),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
        }
    }
}

# Cross-validation configuration
CV_CONFIG = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42,
    'stratify': True,
}

# Hyperparameter optimization configuration
HPO_CONFIG = {
    'n_trials': 200,
    'timeout': None,  # None = no time limit
    'n_jobs': -1,  # Use all cores
    'random_state': 42,
}