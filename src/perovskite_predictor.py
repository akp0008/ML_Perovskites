import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import optuna
import os
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple, Optional

class PerovskitePropertyPredictor:
    """
    A unified ML pipeline for predicting properties of Halide Perovskites.
    
    Features:
    - Agentic Hyperparameter Optimization (Optuna)
    - Gradient Boosting (CatBoost)
    - Explainable AI (SHAP, Feature Importance)
    - Diagnostic Plots (Learning Curves, Parity Plots)
    """

    def __init__(self, data_path: str, target_col: str, output_dir: str = '../results'):
        self.data_path = data_path
        self.target_col = target_col
        self.output_dir = output_dir
        self.model = None
        self.study = None
        
        # State variables for XAI and plotting
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        
        # Mapping for pretty LaTeX labels in plots
        self.feature_map = {
            'Atomization energy (eV/atom)': 'E$_{atomization}$',
            'Relative energy1 (eV/atom)': 'E$_{relative1}$',
            'Relative energy2 (eV/atom)': 'E$_{relative2}$',
            'Density (g/cm^3)': 'Density',
            'rA(Ang)': 'r$_A$', 'rB(Ang)': 'r$_B$', 'rX(Ang)': 'r$_X$'
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads data, filters F==3 (single halide), and cleans columns."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} not found.")
            
        df = pd.read_csv(self.data_path)
        
        # Filter for single-halide (F==3) if column exists
        if 'F' in df.columns:
            df = df[df['F'] != 3].copy().drop('F', axis=1)

        # Drop non-predictive metadata to prevent data leakage
        drop_cols = ['HOIP entry ID', 'Label', 'Dielectric constant, total', 
                     'Refractive index', 'A SITE DFE', 'B SITE DFE', 'X SITE DFE', 
                     'Bandgap, GGA (eV)', 'Volume of the unit cell (A^3)', 
                     'Dielectric constant, electronic', 'Dielectric constant, ionic']
        
        # Keep the target column, drop the rest from X
        X = df.drop([c for c in drop_cols if c in df.columns and c != self.target_col], axis=1)
        
        # If target is in X, separate it
        if self.target_col in X.columns:
            y = X[self.target_col]
            X = X.drop(self.target_col, axis=1)
        else:
            raise ValueError(f"Target {self.target_col} not found in dataset.")
            
        X.rename(columns=self.feature_map, inplace=True)
        self.X, self.y = X, y
        print(f"Data Loaded: {X.shape[0]} samples, {X.shape[1]} features.")
        return X, y

    def objective(self, trial, X_train, y_train):
        """Optuna objective function for hyperparameter tuning."""
        params = {
            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
            'depth': trial.suggest_int('depth', 4, 10),
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'l2_leaf_reg': trial.suggest_categorical('l2_leaf_reg', [1, 3, 5, 10]),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
            'grow_policy': trial.suggest_categorical('grow_policy', ['Depthwise', 'Lossguide']),
            'loss_function': 'RMSE',
            'silent': True,
            'random_state': 42,
            'thread_count': -1
        }
        
        model = CatBoostRegressor(**params)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        return -scores.mean()

    def run_optimization(self, n_trials=20):
        """Runs the Optuna optimization loop."""
        print(f"\n--- Starting Optimization for {self.target_col} ---")
        X_train, _, y_train, _ = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        self.study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        self.study.optimize(lambda t: self.objective(t, X_train, y_train), n_trials=n_trials)
        
        print(f"Best Params: {self.study.best_params}")
        return self.study.best_params

    def train_final_model(self, params=None):
        """Trains the model with best parameters and evaluates it."""
        if not params: 
            params = {'learning_rate': 0.1, 'depth': 6, 'iterations': 1000, 'silent': True}
        
        # Store splits for plotting later
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        self.model = CatBoostRegressor(**params, silent=True)
        self.model.fit(self.X_train, self.y_train)
        
        self.y_pred = self.model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        
        print(f"Final Results for {self.target_col}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R2:  {r2:.4f}")
        
        return r2

    def generate_all_plots(self):
        """Wrapper to generate all diagnostic and explainability plots."""
        if self.model is None:
            raise ValueError("Model not trained. Run train_final_model() first.")
        
        base_filename = self.target_col.split(',')[0].replace(" ", "_") # Clean filename
        
        self.plot_parity(base_filename)
        self.plot_feature_importance(base_filename)
        self.plot_shap_summary(base_filename)
        self.plot_learning_curve_graph(base_filename)
        print(f"All plots saved to {self.output_dir}/")

    def plot_parity(self, filename):
        """Generates a publication-quality parity plot."""
        plt.figure(figsize=(8, 7))
        plt.scatter(self.y_test, self.y_test, color='blue', marker='o', label='Actual Values', alpha=0.6)
        plt.scatter(self.y_test, self.y_pred, color='red', marker='x', label='Predicted Values', alpha=0.8)
        
        plt.title(f'{self.target_col}', fontsize=18, fontweight='bold')
        plt.xlabel('Actual Value', fontsize=14, fontweight='bold')
        plt.ylabel('Predicted Value', fontsize=14, fontweight='bold')
        plt.legend(loc="best", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}_parity.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, filename):
        """Generates a feature importance bar plot."""
        importances = self.model.get_feature_importance()
        indices = np.argsort(importances)[-15:] # Top 15 features
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
        
        plt.barh(range(len(indices)), importances[indices], color=colors)
        plt.yticks(range(len(indices)), [self.X.columns[i] for i in indices], fontsize=12, fontweight='bold')
        plt.title(f'Feature Importance: {self.target_col}', fontsize=18, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=14, fontweight='bold')
        
        # Add value labels
        for index, value in enumerate(importances[indices]):
            plt.text(value, index, f"{value:.2f}", va='center', fontsize=10)
            
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}_importance.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_shap_summary(self, filename):
        """Generates SHAP summary plot."""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.title(f'SHAP Summary: {self.target_col}', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('SHAP Value (Impact on model output)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}_shap.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_learning_curve_graph(self, filename):
        """Generates Learning Curve to diagnose bias/variance."""
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X_train, self.y_train, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5), scoring='r2'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        
        plt.figure(figsize=(10, 7))
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="CV Score")
        
        plt.title(f'Learning Curve: {self.target_col}', fontsize=18, fontweight='bold')
        plt.xlabel('Training Examples', fontsize=14, fontweight='bold')
        plt.ylabel('R2 Score', fontsize=14, fontweight='bold')
        plt.legend(loc="best", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}_learning_curve.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # --- Example Usage ---
    # 1. Setup
    # NOTE: Update 'data/dataset.csv' to the actual location in your repo
    data_file = '../data/dataset.csv'
    
    # 2. Define Targets (Bandgap, Formation Energies, etc.)
    targets = ['Bandgap, GGA (eV)', 'A SITE DFE', 'B SITE DFE']
    
    for target in targets:
        try:
            # Initialize pipeline
            predictor = PerovskitePropertyPredictor(data_file, target_col=target)
            
            # Load Data
            predictor.load_data()
            
            # Agentic Optimization 
            best_params = predictor.run_optimization(n_trials=1000)
            
            # Train Final Model
            predictor.train_final_model(best_params)
            
            # Generate All Visualizations (SHAP, Parity, Importance, Learning Curve)
            predictor.generate_all_plots()
            
        except Exception as e:
            print(f"Skipping {target}: {e}")
