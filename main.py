from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from utils import load_and_preprocess_data, evaluate_function, hyperparameter_tuning


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


def main():
    x_train, x_test, y_train, y_test = load_and_preprocess_data("data/jm1.arff.txt")

    # Select the algorithm for hyperparameter tuning
    algorithms = [
        'decision_tree',
        'random_forest',
        'svm',
        'linear_regression'
    ]

    for algorithm_name in algorithms:
        print(f"Processing {algorithm_name}...")

        # Perform hyperparameter tuning
        best_model = hyperparameter_tuning(x_train, y_train, algorithm_name)

        # Evaluate the best model
        evaluation_results = evaluate_function(best_model, x_train, x_test, y_train, y_test, 0.515)

        # Print evaluation results
        print(f"Results for {algorithm_name}:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value}")
        print("\n")

        # Print top n samples most likely to have defects
        print_top_n_defect_predictions(best_model, x_test, 10)
        print("\n")


if __name__ == "__main__":
    main()
