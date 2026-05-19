from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Imports de Regressão
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class ModelSelector:
    options = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],
                'copy_X': [True, False]
            }
        },
        'ridge': {  # Regressão Linear com regularização L2
            'model': Ridge(),
            'params': {
                'alpha': [0.01, 0.1, 1, 10, 100],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
            }
        },
        'lasso': {  # Regressão Linear com regularização L1
            'model': Lasso(),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1, 10],
                'selection': ['cyclic', 'random']
            }
        },
        'elastic_net': { # Combinação de L1 e L2
            'model': ElasticNet(),
            'params': {
                'alpha': [0.01, 0.1, 1, 10],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        },
        'knn': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 12, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'svr': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto'],
                'epsilon': [0.01, 0.1, 0.5] # Margem de tolerância do erro
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'loss': ['squared_error', 'huber', 'absolute_error']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10]
            }
        }
    }

    def get(self):
        model = Pipeline([
            ('scaler', StandardScaler()), 
            ('model', self.selected['model'])
        ])
        
        raw_params = self.selected['params']
        params = {f'model__{k}': v for k, v in raw_params.items()}
        return model, params

    def __init__(self, name: str):
        self.chosen   = name
        self.selected = self.options[name]