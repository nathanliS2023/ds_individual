from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
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
        'params': {}
    }
}


def get_metrics(y_train, y_test, train_predictions, test_predictions):
    # Classification metrics
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    train_precision = precision_score(y_train, train_predictions)
    test_precision = precision_score(y_test, test_predictions)
    train_recall = recall_score(y_train, train_predictions)
    test_recall = recall_score(y_test, test_predictions)
    train_f1 = f1_score(y_train, train_predictions)
    test_f1 = f1_score(y_test, test_predictions)

    # Confusion matrix for test data
    test_confusion_matrix = confusion_matrix(y_test, test_predictions)

    return {
        'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy,
        'train_precision': train_precision, 'test_precision': test_precision,
        'train_recall': train_recall, 'test_recall': test_recall,
        'train_f1': train_f1, 'test_f1': test_f1,
        'test_confusion_matrix': test_confusion_matrix
    }


def hyperparameter_tuning(x_train, y_train, algorithm_name, model_type, scoring="accuracy"):
    if model_type == 'regression':
        algorithms = regression_algorithms
    elif model_type == 'classification':
        algorithms = classification_algorithms
    else:
        raise ValueError("Invalid model type. Choose 'regression' or 'classification'.")

    if algorithm_name not in algorithms:
        raise ValueError("Invalid algorithm choice.")

    algorithm = algorithms[algorithm_name]
    model = algorithm['model']
    param_grid = algorithm['params']

    print("Doing grid search for", algorithm_name)
    grid_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_


# Evaluate function for regression models with binary outcomes
def evaluate_regression(model, x_train, x_test, y_train, y_test, threshold=0.5):
    # Predict continuous values
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    # Binarize predictions based on the threshold
    binarize = lambda x: np.where(x > threshold, 1, 0)
    binary_train_predictions = binarize(train_predictions)
    binary_test_predictions = binarize(test_predictions)

    # Regression metrics
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)

    metrics = get_metrics(y_train, y_test, binary_train_predictions, binary_test_predictions)
    metrics['train_mse'] = train_mse
    metrics['test_mse'] = test_mse

    return metrics


def evaluate_classification(model, x_train, x_test, y_train, y_test):
    test_predictions = model.predict(x_test)
    train_predictions = model.predict(x_train)

    return get_metrics(y_train, y_test, train_predictions, test_predictions)
