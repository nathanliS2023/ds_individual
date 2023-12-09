from sklearn.model_selection import RandomizedSearchCV

from utils import load_and_preprocess_data, evaluate_regression, evaluate_classification, hyperparameter_tuning, \
    plot_model_metrics
from models import get_nn_model
from partical_swarm import particle_swarm_optimization


def analyze_models(models, x_train, x_test, y_train, y_test, model_type):
    scoring = "neg_mean_squared_error" if model_type == 'regression' else "accuracy"
    evaluate = evaluate_regression if model_type == 'regression' else evaluate_classification

    model_metrics = {}

    for model_name in models:
        print(f"Processing {model_name}...")

        # Special handling for neural network
        if model_name == 'neural_network':
            model = get_nn_model(input_dim=x_train.shape[1])
            param_grid = {
                'batch_size': [10, 20, 50],
                'epochs': [10, 50, 100],
                'learning_rate': [0.001, 0.01, 0.1],
            }
            grid_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring=scoring, n_jobs=-1)
            grid_search.fit(x_train, y_train, validation_split=0.2)
            best_model = grid_search.best_estimator_
        else:
            # Perform hyperparameter tuning for other models
            best_model = hyperparameter_tuning(x_train, y_train, model_name, model_type, scoring=scoring)

        # Evaluate the best model
        evaluation_results = evaluate(best_model, x_train, x_test, y_train, y_test)

        model_metrics[model_name] = evaluation_results

        # Print evaluation results
        print(f"Results for {model_name}:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value}")
        print("\n")

    plot_model_metrics(model_metrics)


def main():
    x_train, x_test, y_train, y_test = load_and_preprocess_data("data/jm1.arff.txt")

    # Select the algorithm for hyperparameter tuning
    regression_models = [
        'decision_tree_reg',
        'random_forest_reg',
        'svr',
        'linear_regression',
        'gradient_boosting_reg',
        'ridge_regression',
    ]

    classification_models = [
        'decision_tree_clf',
        'random_forest_clf',
        'svc',
        'logistic_regression',
        'naive_bayes',
        'neural_network',
    ]

    # Perform feature selection
    num_features = x_train.shape[1]
    num_particles = 10
    num_iterations = 20
    best_feature_subset = particle_swarm_optimization(x_train, y_train, num_features, num_particles, num_iterations)
    print(best_feature_subset)

    x_train = x_train.iloc[:, best_feature_subset]
    x_test = x_test.iloc[:, best_feature_subset]

    # Evaluate the models
    analyze_models(classification_models, x_train, x_test, y_train, y_test, model_type='classification')


if __name__ == "__main__":
    main()
