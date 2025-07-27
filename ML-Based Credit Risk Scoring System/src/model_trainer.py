from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier()
        }
        self.best_model = None
        
    def train_models(self, X_train, y_train):
        """Train multiple models and select the best one"""
        best_score = 0
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
            
            if score > best_score:
                best_score = score
                self.best_model = (name, model)
    
    def predict_risk(self, X):
        """Predict credit risk and return risk category"""
        if self.best_model is None:
            raise Exception("Model not trained yet")
            
        probabilities = self.best_model[1].predict_proba(X)
        predictions = self.best_model[1].predict(X)
        
        risk_categories = []
        for prob in probabilities:
            if prob[1] < 0.3:
                risk_categories.append('Low Risk ✅')
            elif prob[1] < 0.7:
                risk_categories.append('Medium Risk ⚠️')
            else:
                risk_categories.append('High Risk ❌')
                
        return predictions, risk_categories
    
    def save_model(self, path):
        """Save the best model to disk"""
        if self.best_model is None:
            raise Exception("No model to save")
        joblib.dump(self.best_model[1], path)