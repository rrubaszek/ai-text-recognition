from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
        
from models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, hyperparams=None, **kwargs):
        super().__init__(hyperparams, **kwargs)
        self.scaler = StandardScaler()
        self._build_model()
    
    def _build_model(self):
        params = self.hyperparams.copy()
        params.setdefault('random_state', self.random_state)
        self.model = LogisticRegression(**params)
    
    def train(self, X_train, y_train, **kwargs):
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train, **kwargs)
        self.is_fitted = True
    
    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_model_name(self):
        return "LogisticRegression"