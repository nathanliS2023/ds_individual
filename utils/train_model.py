from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import numpy as np

# Dictionary mapping algorithms to their models and parameter grids
regression_algorithms = {
    'decision_tree_reg': {
        'model': DecisionTreeRegressor(),
        'params': {
            'max_depth': [None, 10, 15, 20, 25],
            'min_samples_leaf': [1, 2, 5, 10]
        }
    },
    'random_forest_reg': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 15, 20, 25],
            'min_samples_leaf': [1, 2, 5, 10]
        }
    },
    'svr': {
        'model': SVR(),
        'params': {
            'C': [0.01, 0.1, 1, 10],
            'gamma': ['scale', 'auto'],
        }
    },
    'linear_regression': {
        'model': LinearRegression(),
        'params': {}  # Typically does not have hyperparameters for grid search
    },
    'gradient_boosting_reg': {
        'model': GradientBoostingRegressor(),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    },
    'ridge_regression': {
        'model': Ridge(),
        'params': {
            'alpha': [0.01, 0.1, 1, 10],
        }
    }
}

classification_algorithms = {
    'decision_tree_clf': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [None, 10, 15, 20, 25],
            'min_samples_leaf': [1, 2, 5, 10]
        }
    },
    'random_forest_clf': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 15, 20, 25],
            'min_samples_leaf': [1, 2, 5, 10]
        }
    },
    'svc': {
        'model': SVC(),
        'params': {
            'C': [0.01, 0.1, 1, 10],
            'gamma': ['scale', 'auto'],
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'max_iter': [1000],
        }
    },
    'naive_bayes': {
        'model': GaussianNB(),
        'params': {}
    }
}


def hyperparameter_tuning(x_train, y_train, algorithm_name, model_type, scoring="accuracy"):
    if model_type == 'regression':
        algorithms = regression_algorithms
    elif model_type == 'classification':
        algorithms = classification_algorithms
    else:
        raise ValueError("Invalid model type. Choose 'regression' or 'classification'.")

    if algorithm_name not in algorithms:
        raise ValueError("Invalid algorithm choice:" + str(algorithm_name))

    algorithm = algorithms[algorithm_name]
    model = algorithm['model']
    param_grid = algorithm['params']

    print("Doing grid search for", algorithm_name)
    grid_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_
