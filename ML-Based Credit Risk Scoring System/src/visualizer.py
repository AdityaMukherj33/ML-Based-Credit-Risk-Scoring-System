import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class Visualizer:
    def plot_feature_importance(self, model, X, feature_names):
        """Plot feature importance using SHAP values"""
        # Convert X to numpy array if it's not already
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
            
        # Ensure feature_names matches the number of features
        if len(feature_names) != X.shape[1]:
            raise ValueError(f"Number of feature names ({len(feature_names)}) does not match number of features in X ({X.shape[1]})")
            
        # Choose appropriate explainer based on model type
        if isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, take positive class
        else:
            # For non-tree models like LogisticRegression, use KernelExplainer
            # Use a smaller sample size for better performance
            sample_size = min(20, X.shape[0])
            background = shap.sample(X, sample_size)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X[:sample_size])[1]
            X = X[:sample_size]  # Match X to the number of samples we explained
        
        plt.figure(figsize=(10, 6))
        # Calculate mean absolute SHAP values for each feature
        mean_shap = np.abs(shap_values).mean(0)
        # Get indices of features sorted by importance
        feature_indices = np.argsort(-mean_shap)
        # Select top features (use all if less than 2 features)
        n_features = min(2, len(feature_names))
        selected_indices = feature_indices[:n_features]
        
        # Select corresponding feature names and data
        selected_features = [feature_names[i] for i in selected_indices]
        selected_shap_values = shap_values[:, selected_indices]
        selected_X = X[:, selected_indices]
        
        shap.summary_plot(
            selected_shap_values,
            selected_X,
            feature_names=selected_features,
            show=False
        )
        plt.tight_layout()
        plt.show()
        
    def plot_risk_distribution(self, risk_categories):
        """Plot distribution of risk categories"""
        plt.figure(figsize=(8, 6))
        sns.countplot(x=risk_categories)
        plt.title('Distribution of Risk Categories')
        plt.show()