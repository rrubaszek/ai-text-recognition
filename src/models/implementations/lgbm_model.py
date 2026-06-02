import lightgbm as lgb

from models.base_model import BaseModel

class LGBMClassifierModel(BaseModel):
    def __init__(self, hyperparams=None, **kwargs):
        super().__init__(hyperparams, **kwargs)
        self._build_model()
    
    def _build_model(self):
        params = self.hyperparams.copy()
        params.setdefault('random_state', self.random_state)
        self.model = lgb.LGBMClassifier(**params)
    
    def train(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train, **kwargs)
        self.is_fitted = True
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_model_name(self):
        return "LightGBM"