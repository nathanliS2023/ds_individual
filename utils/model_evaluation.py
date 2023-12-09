import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def plot_model_metrics(model_metrics):
    # Extracting the required metrics
    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    labels = list(model_metrics.keys())
    data = {metric: [model_metrics[label][metric] for label in labels] for metric in metrics}

    # Creating the bar chart
    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, data[metric], width, label=metric)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by model and metric')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


def get_metrics(y_train, y_test, train_predictions, test_predictions):
    # Classification metrics
    # train_accuracy = accuracy_score(y_train, train_predictions)
    # train_precision = precision_score(y_train, train_predictions)
    # train_recall = recall_score(y_train, train_predictions)
    # train_f1 = f1_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)

    # Confusion matrix for test data
    test_confusion_matrix = confusion_matrix(y_test, test_predictions)

    return {
        # 'train_accuracy': train_accuracy,
        # 'train_precision': train_precision,
        # 'train_recall': train_recall,
        # 'train_f1': train_f1,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_confusion_matrix': test_confusion_matrix
    }


def binarize(x, threshold=0.5):
    return np.where(x > threshold, 1, 0)


# Evaluate function for regression models with binary outcomes
def evaluate_regression(model, x_train, x_test, y_train, y_test, threshold=0.5):
    # Predict continuous values
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    # Binarize predictions based on the threshold
    binary_train_predictions = binarize(train_predictions)
    binary_test_predictions = binarize(test_predictions)

    metrics = get_metrics(y_train, y_test, binary_train_predictions, binary_test_predictions)

    return metrics


def evaluate_classification(model, x_train, x_test, y_train, y_test):
    test_predictions = model.predict(x_test)
    train_predictions = model.predict(x_train)

    return get_metrics(y_train, y_test, train_predictions, test_predictions)


def evaluate_feature_subset(X, y, selected_features):
    # Select only the features marked as True in selected_features
    X_subset = X.iloc[:, selected_features]

    # Define the model to use for evaluation
    model = RandomForestClassifier()

    # Perform cross-validation and return the mean score
    scores = cross_val_score(model, X_subset, y, cv=5, scoring='accuracy')
    print(scores)

    return np.mean(scores)