from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from utils import load_and_preprocess_data, evaluate_function, hyperparameter_tuning


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
        evaluation_results = evaluate_function(best_model, x_train, x_test, y_train, y_test)

        # Print evaluation results
        print(f"Results for {algorithm_name}:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value}")
        print("\n")


if __name__ == "__main__":
    main()
