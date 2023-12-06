from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from utils import load_and_preprocess_data, evaluate_regression, evaluate_classification, hyperparameter_tuning


def print_top_n_defect_predictions(model, x_test, n):
    # Ensure the model has a predict_proba method
    if hasattr(model, 'predict_proba'):
        # Get probability estimates
        proba = model.predict_proba(x_test)

        # Assuming the positive (defect) class is the second column
        defect_proba = proba[:, 1]

        # Get the indices of the top n predictions
        top_n_indices = defect_proba.argsort()[-n:][::-1]

        # Print the results
        print(f"Top {n} samples most likely to have defects:")
        for index in top_n_indices:
            print(f"Sample {index} - Probability of defect: {defect_proba[index]}")
    else:
        print("Model does not support probability predictions.")


def analyze_models(models, x_train, x_test, y_train, y_test, model_type):
    scoring = None
    evaluate = None
    if model_type == 'regression':
        scoring = "neg_mean_squared_error"
        evaluate = evaluate_regression
    elif model_type == 'classification':
        scoring = "accuracy"
        evaluate = evaluate_classification

    for model in models:
        print(f"Processing {model}...")

        # Perform hyperparameter tuning
        best_model = hyperparameter_tuning(x_train, y_train, model, model_type, scoring=scoring)

        # Evaluate the best model
        evaluation_results = evaluate(best_model, x_train, x_test, y_train, y_test)

        # Print evaluation results
        print(f"Results for {model}:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value}")
        print("\n")

        # Print top n samples most likely to have defects
        # print_top_n_defect_predictions(best_model, x_test, 10)
        # print("\n")


def main():
    x_train, x_test, y_train, y_test = load_and_preprocess_data("data/jm1.arff.txt")

    # Select the algorithm for hyperparameter tuning
    regression_models = [
        'decision_tree_reg',
        'random_forest_reg',
        'svr',
        'linear_regression',
    ]

    classification_models = [
        'decision_tree_clf',
        'random_forest_clf',
        'svc',
        'logistic_regression'
    ]

    # Evaluate the models
    analyze_models(regression_models, x_train, x_test, y_train, y_test, model_type='regression')


if __name__ == "__main__":
    main()
